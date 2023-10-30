classdef TrainingManager < handle
    % TRAININGMANAGER
    
    % Copyright 2018-2020 The MathWorks, Inc.
    
    properties
        Agents
        Environment
        TrainingOptions
    end
    properties (Hidden, SetAccess = private)
        
        HasCritic (1,:) logical
        
        % episode count "states"
        EpisodeCount = 0
        TotalEpisodeStepCount = 0
        
        % window "states"
        RewardsInAveragingWindow
        StepsInAveragingWindow
        
        % construct training result struct
        TrainingStartTime
        TrainingElapsedTime
        Watch
        Devices
        LearnRates
        
        % simulation info
        SimulationInfo = {}
        
        TrainingStats
        
        % Training status:
        % 0 : Agent has met a stop training criteria
        % 1 : Agent is training
        TrainingStatus
        
        TrainingStoppedReason
        TrainingStoppedValue
    end
    properties (Access = private,Transient)
        % listener to determine if stop training has been requested
        StopTrainingListener
        
        % listener for episode finished event
        EpisodeFinishedListener
        
        % listener for data rcv on worker
        DataReceivedOnWorkerListener
        
        % episode manager
        EpisodeMgr
        
        % listeners for tasks on workers
        TaskListeners
        
        % pre train settings
        PreTrainAgentSettings = []
        
    end
    events
        DataReceivedFromWorker
        TrainingManagerUpdated
    end
    methods
        function this = TrainingManager(env,agents,opt)
            this.Environment     = env;
            this.Agents          = agents;
            this.TrainingOptions = opt;
            this.HasCritic       = arrayfun(@(a) hasCritic(a),this.Agents);
            this.TrainingStatus  = ones(1,numel(this.Agents));
            this.TrainingStoppedReason = repmat("",1,numel(this.Agents));
            this.TrainingStoppedValue  = repmat("",1,numel(this.Agents));
            
            % watch
            this.Watch = nnet.internal.cnn.ui.adapter.Stopwatch();
            reset(this.Watch);
            
            % training start time
            this.TrainingStartTime = iDateTimeAsStringUsingDefaultLocalFormat(datetime('now'));
        end
        function delete(this)
            delete(this.StopTrainingListener);
            delete(this.EpisodeFinishedListener);
            delete(this.DataReceivedOnWorkerListener);
            delete(this.TaskListeners);
        end
        function cleanup(this)
            arrayfun(@(a) setStepMode(a,"sim"),this.Agents);
            % only call postTrain if preTrain was successfully called,
            % which will return a struct.
            for idx = 1:getNumAgents(this)
                if ~isempty(this.PreTrainAgentSettings) && ~isempty(this.PreTrainAgentSettings{idx})
                    postTrain(this.Agents(idx),this.PreTrainAgentSettings{idx});
                end
            end
            this.PreTrainAgentSettings = [];
        end
        function setActionMessage(this,msg)
            arguments
                this
                msg string {mustBeTextScalar}
            end
            % set an action message for the episode manager
            if isvalid(this)
                episodeMgr = this.EpisodeMgr;
                if ~isempty(episodeMgr) && isvalid(episodeMgr)
                    setActionMessage(episodeMgr,msg);
                end
            end
        end
        function msg = getActionMessage(this,id)
            arguments
                this
                id double = 1
            end
            % get an action message from the episode manager
            msg = '';
            if isvalid(this)
                episodeMgr = this.EpisodeMgr;
                if ~isempty(episodeMgr)
                    msg = getActionMessage(episodeMgr,id);
                end
            end
        end
        function reqSimulink = requiresSimulink(this)
            % is simulink needed to run the training
            reqSimulink = isa(this.Environment,'rl.env.SimulinkEnvWithAgent');
        end
        function globalStopTraining = update(this,episodeFinishedInfo)
            % update the manager once an episode finishes
            
            epinfo       = episodeFinishedInfo.EpisodeInfo   ;
            episodeCount = episodeFinishedInfo.EpisodeCount  ;
            
            workerID     = episodeFinishedInfo.WorkerID      ;
            simInfo      = episodeFinishedInfo.SimulationInfo;
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % update the episode number in workspace
            assignin("base", "EpiCount", episodeCount);
            assignin("base","instantTrainingStats",this.TrainingStats);

%             xy_x = simInfo.xy.Data(:,1);
%             epiEd = simInfo.energy.Data(end)/xy_x(end);
%             epiEdList = evalin("base", 'epiEdList');
%             epiEdList(find(epiEdList,1,'Last')+1)=epiEd;
%             assignin("base", "epiEdList", epiEdList);

%             epiRunTimeList = evalin("base", 'epiRunTimeList');
%             epiRunTimeList(find(epiRunTimeList,1,'Last')+1)=simInfo.energy.Time(end);
%             assignin("base", "epiRunTimeList", epiRunTimeList);
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

            numAgents = getNumAgents(this);
            
            % evaluate q0 on the agent
            for idx = 1:numAgents
                if epinfo(idx).StepsTaken > 0
                    q0 = evaluateQ0(this.Agents(idx),epinfo(idx).InitialObservation);
                else
                    q0 = 0;
                end
                epinfo(idx).Q0 = q0;
            end
            
            % attach the episode info
            this.SimulationInfo{episodeCount} = simInfo;
            
            % update "states"
            this.EpisodeCount          = episodeCount;
            this.TotalEpisodeStepCount = this.TotalEpisodeStepCount + [epinfo.StepsTaken];
            
            % compute info and update displays
            info                = computeMetrics(this,epinfo);
            globalStopTraining  = updateDisplaysFromTrainingInfo(this,info);
            
            % ----------- AGENT STOP TRAINING ------------
            %
            % When an agent has reached stop training value,
            %     1. log training stopped reason and value
            %     2. set step mode to sim
            %     3. update episode manager for those agents
            
            % determine stop training
            stopCriteria = this.TrainingOptions.StopTrainingCriteria;
            stopFlag    = this.TrainingOptions.StopTrainingFunction(info);
            if strcmpi(stopCriteria,'Custom') && isscalar(stopFlag)
                stopFlag = repmat(stopFlag,1,numAgents); % perform scalar expansion
            end
            
            currentStatus = this.TrainingStatus;
            
            for idx = 1:numAgents
                if currentStatus(idx) && stopFlag(idx)  % stop training true
                    % update training status
                    this.TrainingStatus(idx) = 0;
                    
                    % log reason and value
                    if strcmpi(stopCriteria,'Custom')
                        % for custom stop criteria
                        reason = ['@' func2str(this.TrainingOptions.StopTrainingValue)];
                        value = stopFlag(idx);
                    else
                        reason = stopCriteria;
                        value  = this.TrainingOptions.StopTrainingValue(idx);
                    end
                    this.TrainingStoppedReason(idx) = reason;
                    this.TrainingStoppedValue(idx)  = string(value);
                    
                    % change step mode to sim
                    setStepMode(this.Agents(idx),"sim");
                    
                    % update episode manager
                    if ~isempty(this.EpisodeMgr) && isvalid(this.EpisodeMgr)
                        stopTrainingAgent(this.EpisodeMgr,idx,reason,value);
                    end
                end
            end
            
            % terminate the simulation if criteria is met
            terminateSimulation(this.Environment,globalStopTraining);
            
            % tell the world that the training manager has been updated
            s                     = struct;
            s.EpisodeInfo         = epinfo;
            s.ComputedEpisodeInfo = info;
            s.EpisodeCount        = episodeCount;
            s.WorkerID            = workerID;
            ed                    = rl.util.RLEventData(s);
            notify(this,'TrainingManagerUpdated',ed);
        end
        function stats = run(this)
            % run the training routine with setup and cleanup
            clnup = onCleanup(@() cleanup(this));
            
            preTrain(this);
            train(this);
            stats = postTrain(this);
        end
        function preTrain(this)
            % run before training occurs
            
            % make sure the agent is reset
            env = this.Environment;
            agents = this.Agents;
            numAgents = getNumAgents(this);
            for agentIdx = 1 : numAgents
                agent = agents(agentIdx);
                agent.MaxSteps = this.TrainingOptions.MaxStepsPerEpisode;
            end
            % To support heterogeneous array case, store pre train agent
            % settings in a cell array
            this.PreTrainAgentSettings = arrayfun(@(a) preTrain(a), agents, 'UniformOutput',false);
            terminateSimulation(env,false);
            
            % perform scalar expansion of window length, stop value and
            % save value
            numAgents = getNumAgents(this);
            if numAgents > 1
                if numel(this.TrainingOptions.ScoreAveragingWindowLength) == 1
                    this.TrainingOptions.ScoreAveragingWindowLength = ...
                        repmat(this.TrainingOptions.ScoreAveragingWindowLength,1,numAgents);
                end
                if ~strcmpi(this.TrainingOptions.StopTrainingCriteria,'Custom') && numel(this.TrainingOptions.StopTrainingValue) == 1
                    this.TrainingOptions.StopTrainingValue = ...
                        repmat(this.TrainingOptions.StopTrainingValue,1,numAgents);
                end
                if ~strcmpi(this.TrainingOptions.SaveAgentCriteria,'Custom') && numel(this.TrainingOptions.SaveAgentValue) == 1
                    this.TrainingOptions.SaveAgentValue = ...
                        repmat(this.TrainingOptions.SaveAgentValue,1,numAgents);
                end
            end
            
            % initialize the window "states"
            this.RewardsInAveragingWindow = cell(1,numAgents);
            this.StepsInAveragingWindow = cell(1,numAgents);
            for agentIdx = 1:numAgents
                numScoresToAverage = this.TrainingOptions.ScoreAveragingWindowLength(agentIdx);
                this.RewardsInAveragingWindow{agentIdx} = zeros(numScoresToAverage,1,'single');
                this.StepsInAveragingWindow{agentIdx}   = zeros(numScoresToAverage,1,'single');
            end
            
            % build the train stats struct
            maxEpisodes = this.TrainingOptions.MaxEpisodes;
            this.SimulationInfo = cell(1,maxEpisodes);
            % build the saved agents directory
            if isempty(dir(this.TrainingOptions.SaveAgentDirectory))
                if ~strcmpi(this.TrainingOptions.SaveAgentCriteria,"none")
                    try
                        mkdir(this.TrainingOptions.SaveAgentDirectory);
                    catch ex
                        me = MException(message('rl:general:TrainingManagerUnableToCreateSaveDir',this.TrainingOptions.SaveAgentDirectory));
                        throw(addCause(me,ex));
                    end
                end
            end
            
            % Initialize training statistics
            TrainingStatistics = struct([]);
            for agentIndex = 1:numAgents
                TrainingStatistics(agentIndex).TimeStamp       = repmat("",maxEpisodes,1);
                TrainingStatistics(agentIndex).EpisodeIndex     = zeros(maxEpisodes,1);
                TrainingStatistics(agentIndex).EpisodeReward    = zeros(maxEpisodes,1);
                TrainingStatistics(agentIndex).EpisodeSteps     = zeros(maxEpisodes,1);
                TrainingStatistics(agentIndex).AverageReward    = zeros(maxEpisodes,1);
                TrainingStatistics(agentIndex).AverageSteps     = zeros(maxEpisodes,1);
                TrainingStatistics(agentIndex).TotalAgentSteps  = zeros(maxEpisodes,1);
                TrainingStatistics(agentIndex).Information      = [];
                if hasCritic(agents(agentIndex))
                    TrainingStatistics(agentIndex).EpisodeQ0    = zeros(maxEpisodes,1);
                end
            end
            this.TrainingStats = TrainingStatistics;
            
            % create the episode manager
            initializeEpisodeManager(this);
        end
        
        function stats = postTrain(this)
            % return the training stats post train
            
            episodeIndex = this.EpisodeCount;
            this.TrainingElapsedTime = char(getDurationSinceReset(this.Watch));
            episodeMgr = this.EpisodeMgr;
            
            activeAgents = find(this.TrainingStatus~=0);
            
            % determine stop training reason
            % longReason updates final results label
            % stopReason and stopValue updates table in more details dialog
            if isempty(activeAgents)
                % training stopped with all agent reaching stop criteria
                longReason = string(message('rl:general:TextStopCriteriaLongReason'));
                
            elseif episodeIndex >= this.TrainingOptions.MaxEpisodes
                % training stopped with max episodes condition
                longReason = string(message('rl:general:TextMaxEpisodesLongReason'));
                stopReason = string(message('rl:general:TextMaxEpisodes'));
                stopValue = string(message('rl:general:TextEpisode')) + " " + episodeIndex;
                
            else
                % user clicked stop training button
                longReason = string(message('rl:general:TextStopTrainLongReason'));
                stopReason = string(message('rl:general:TextStopButton'));
                stopValue = string(message('rl:general:TextEpisode')) + " " + episodeIndex;
                
            end
            
            % stop training active agents
            for idx = 1:numel(activeAgents)
                if ~isempty(episodeMgr) && isvalid(episodeMgr)
                    stopTrainingAgent(episodeMgr,activeAgents(idx),stopReason,stopValue);
                end
                this.TrainingStoppedReason(activeAgents(idx)) = stopReason;
                this.TrainingStoppedValue(activeAgents(idx))  = string(stopValue);
            end
            
            % set the episode manager to stop training state
            if ~isempty(episodeMgr) && isvalid(episodeMgr)
                stopTraining(episodeMgr,longReason);
            end
            
            % Clean up unused training statistics
            stats = cleanupTrainingStats(this);
            
            % create training result struct for analysis and recreating plot
            trainingInfoStruct = createTrainingInfoStruct(this,longReason);
            for idx = 1:getNumAgents(this)
                stats(idx).Information = trainingInfoStruct(idx);
            end
            
        end
        
        function stats = cleanupTrainingStats(this)
            maxEpisodes = this.TrainingOptions.MaxEpisodes;
            episodeIndex = this.EpisodeCount;
            % Clean up unused training statistics
            rmidx = (episodeIndex+1):maxEpisodes;
            stats = this.TrainingStats;
            for agentIndex = 1 : getNumAgents(this)
                if ~isempty(stats)
                    stats(agentIndex).EpisodeIndex   (rmidx) = [];
                    stats(agentIndex).EpisodeReward  (rmidx) = [];
                    stats(agentIndex).EpisodeSteps   (rmidx) = [];
                    stats(agentIndex).AverageReward  (rmidx) = [];
                    stats(agentIndex).TotalAgentSteps(rmidx) = [];
                    stats(agentIndex).AverageSteps   (rmidx) = [];
                    if hasCritic(this.Agents(agentIndex))
                        stats(agentIndex).EpisodeQ0  (rmidx) = [];
                    end
                    % attach the simulation info to the output structure
                    stats(agentIndex).SimulationInfo = this.SimulationInfo;
                    stats(agentIndex).SimulationInfo((episodeIndex+1):end) = [];
                end
                stats(agentIndex).SimulationInfo = vertcat(this.SimulationInfo{:});
            end
        end
        
        function trainingInfoStruct = createTrainingInfoStruct(this,finalResultText)
            % create training info struct for analysis and plots
            % trainingInfoStruct is a 1xN struct array, where N = number of
            % trained agents.
            numAgents = getNumAgents(this);
            agentName = createUniqueAgentNames(this);
            for idx = numAgents:-1:1
                trainingInfoStruct(idx).EnvironmentName = string(getNameForEpisodeManager(this.Environment));
                trainingInfoStruct(idx).AgentName = agentName(idx);
                if isa(this.Environment,'AgentBlock')
                    trainingInfoStruct(idx).BlockPath = this.Environment.AgentBlock(idx);
                else
                    trainingInfoStruct(idx).BlockPath = [];
                end
                trainingInfoStruct(idx).TrainingOpts = this.TrainingOptions;
                trainingInfoStruct(idx).HasCritic = this.HasCritic(idx);
                trainingInfoStruct(idx).HardwareResource = this.Devices(idx);
                trainingInfoStruct(idx).LearningRate = this.LearnRates{idx};
                trainingInfoStruct(idx).TrainingStartTime = this.TrainingStartTime;
                trainingInfoStruct(idx).ElapsedTime = this.TrainingElapsedTime;
                trainingInfoStruct(idx).TimeStamp = this.TrainingStats(idx).TimeStamp;
                trainingInfoStruct(idx).StopTrainingCriteria = this.TrainingStoppedReason(idx);
                trainingInfoStruct(idx).StopTrainingValue = this.TrainingStoppedValue(idx);
                trainingInfoStruct(idx).FinalResult = finalResultText;
            end
        end
        
        function train(this)
            % train the agent
            
            % create the trainer
            trainer = rl.train.createTrainerFactory(this.Environment,this.Agents,this.TrainingOptions);
            
            % attach the trainer to the training manager
            attachTrainer(this,trainer);
            % on cleanup, detatch the trainer
            cln = onCleanup(@() detatchTrainer(this,trainer));
            % run the trainer
            run(trainer);
        end
        function attachTrainer(this,trainer)
            % attach the training manager to a trainer
            
            this.TaskListeners    = addlistener(trainer,'TasksRunningOnWorkers',...
                @(src,ed) setActionMessage(this,getString(message(...
                'rl:general:TrainingManagerRunningTasksOnWorkers'))));
            this.TaskListeners(2) = addlistener(trainer,'TasksCleanedUpOnWorkers',...
                @(src,ed) setActionMessage(this,getString(message(...
                'rl:general:TrainingManagerCleaningUpWorkers'))));
            this.TaskListeners(3) = addlistener(trainer,'ActionMessageReceived',...
                @(src,ed) setActionMessage(this,ed.Data));
            
            % set the update fcn here (listeners will drop events if not
            % marked as recursive)
            trainer.FinishedEpisodeFcn = @(info) update(this,info);
        end
        function detatchTrainer(this,trainer)
            % detatch the trainer from the training manager
            delete(trainer);
            delete(this.TaskListeners);
        end
        function n = getNumAgents(this)
            n = numel(this.Agents);
        end
    end
    methods (Access = private)
        function attachEnvEpisodeFinishedListener(this)
            this.EpisodeFinishedListener = addlistener(this.Environment,'EpisodeFinished',...
                @(src,ed) update(this,ed.Data));
        end
        function initializeEpisodeManager(this)
            try
                %% get device and learning rate
                % REVISIT: agent abstraction does NOT define
                % getAction/getCritic
                % actor
                numAgents = getNumAgents(this);
                devices    = struct;
                learnRates = cell(1,numAgents);
                for agentIndex = 1 : numAgents
                    actor  = getActor(this.Agents(agentIndex));
                    critic = getCritic(this.Agents(agentIndex));
                    % actor
                    if ~isempty(actor)
                        actorOptions   = actor.Options;
                        actorDevice    = actorOptions.UseDevice;
                        actorLearnRate = actorOptions.LearnRate;
                    end
                    % critic
                    if ~isempty(critic)
                        criticOptions   = critic.Options;
                        criticDevice    = criticOptions.UseDevice;
                        criticLearnRate = criticOptions.LearnRate;
                    end
                    % three cases: actor only, critic only, both actor and critic
                    if ~isempty(actor) && ~isempty(critic)
                        devices(agentIndex).actorDevice  = actorDevice;
                        devices(agentIndex).criticDevice = criticDevice;
                        learnRates{agentIndex}  = [actorLearnRate,criticLearnRate];
                    elseif ~isempty(actor) && isempty(critic)
                        devices(agentIndex).actorDevice  = actorDevice;
                        learnRates{agentIndex}  = actorLearnRate;
                    elseif ~isempty(critic) && isempty(actor)
                        devices(agentIndex).criticDevice = criticDevice;
                        learnRates{agentIndex}  = criticLearnRate;
                    end
                end
            catch
                learnRates = {1};
                devices = struct('criticDevice','unknown');
            end
            this.Devices = devices;
            this.LearnRates = learnRates;
            %% build the episode manager
            if strcmp(this.TrainingOptions.Plots,'training-progress')
                delete(this.StopTrainingListener);
                
                envName      = string(getNameForEpisodeManager(this.Environment));
                agentName    = createUniqueAgentNames(this);
                trainOptions = this.TrainingOptions;
                dataName = [...
                    "EpisodeReward";
                    "EpisodeSteps";
                    "GlobalStepCount";
                    "AverageReward";
                    "AverageSteps";
                    "EpisodeQ0" ];
                displayName = [...
                    string(message('rl:general:TextEpisodeReward'));
                    string(message('rl:general:TextEpisodeSteps'));
                    string(message('rl:general:TextTotalNumSteps'));
                    string(message('rl:general:TextAverageReward'));
                    string(message('rl:general:TextAverageSteps'));
                    string(message('rl:general:TextEpisodeQ0')) ];
                emOptions = rl.internal.episodeManager.util.EpisodeManagerOptions( ...
                    agentName, dataName, ...
                    'EnvName', envName, ...
                    'DisplayName', displayName, ...
                    'TrainingOptions', trainOptions, ...
                    'HasCritic',this.HasCritic, ...
                    'ShowOnFigure', [1,0,0,1,0,1], ...
                    'Color',["#B0E2FF","k","k","#0072BD","k","#EDB120"] );
                if isa(this.Environment,'rl.env.SimulinkEnvWithAgent')
                    blk = this.Environment.AgentBlock;
                    emOptions.AgentBlock = blk;
                end
                if isempty(trainOptions.View)
                    % launch standalone
                    episodeMgr = rl.internal.episodeManager.EpisodeManager(emOptions);
                else
                    % launch embedded in RL app
                    view = trainOptions.View;
                    episodeMgr = rl.internal.episodeManager.EpisodeManager(emOptions,view);
                end
                
                % bridge request to terminate simulations from the episode
                % manager to the environment
                this.StopTrainingListener = addlistener(episodeMgr,...
                    'RequestToStopTraining','PostSet',...
                    @(src,ed) request2ManuallyTerminateCB(this,src,ed));
                
                % store the episode manager
                this.EpisodeMgr = episodeMgr;
                setStartTime(this.EpisodeMgr,this.TrainingStartTime);
            end
        end
        function info = computeMetrics(this,epinfo)
            % returns relevant training progress metrics as a struct info.
            %
            %  Info.AverageSteps   : Running average of number of steps per episode
            %  Info.AverageReward  : Running average of reward per episode
            %  Info.EpisodeReward  : Reward for current episode
            %  Info.GlobalStepCount: Total times the agent was invoked
            %  Info.EpisodeCount   : Total number of episodes the agent has trained for
            %  Info.TrainingStatus : Training status of agents (0 or 1)
            
            episodeIndex = this.EpisodeCount;
            episodeSteps = [epinfo.StepsTaken];
            episodeReward = [epinfo.CumulativeReward];
            totalStepCount = this.TotalEpisodeStepCount;
            q0 = [epinfo.Q0];
            
            % circular buffer index for averaging window
            numAgents = getNumAgents(this);
            for agentIdx = numAgents:-1:1
                numScoresToAverage = this.TrainingOptions.ScoreAveragingWindowLength(agentIdx);
                idx = mod(episodeIndex-1,numScoresToAverage)+1;
                this.RewardsInAveragingWindow{agentIdx}(idx) = episodeReward(agentIdx);
                this.StepsInAveragingWindow{agentIdx}(idx)   = episodeSteps(agentIdx);
                numScores = min(episodeIndex,numScoresToAverage);
                avgReward(agentIdx) = sum(this.RewardsInAveragingWindow{agentIdx})/numScores;
                avgSteps(agentIdx) = sum(this.StepsInAveragingWindow{agentIdx})/numScores;
            end
            
            info.AverageSteps    = avgSteps;
            info.AverageReward   = avgReward;
            info.EpisodeReward   = episodeReward;
            info.GlobalStepCount = totalStepCount;
            info.EpisodeCount    = episodeIndex;
            info.EpisodeSteps    = episodeSteps;
            if any(this.HasCritic)
                info.EpisodeQ0 = q0;
            end
            info.TrainingStatus = this.TrainingStatus;
        end
        function updateCommandLineDisplay(this,info)
            % update the command line display
            
            if this.TrainingOptions.Verbose
                episodeIndex = info.EpisodeCount;
                MaxEpisodes     = this.TrainingOptions.MaxEpisodes;
                stepCount       = info.EpisodeSteps;
                globalStepCount = info.GlobalStepCount;
                
                episodeText = getString(message('rl:general:TextEpisode'));
                episodeRewardText = getString(message('rl:general:TextEpisodeReward'));
                episodeStepsText = getString(message('rl:general:TextEpisodeSteps'));
                averageRewardText = getString(message('rl:general:TextAverageReward'));
                episodeQ0Text = getString(message('rl:general:TextEpisodeQ0'));
                stepCountText = getString(message('rl:general:TextStepCount'));
                
                numAgents = getNumAgents(this);
                if numAgents > 1
                    % if multi agent case, display stats in a table format
                    fprintf('%s: %3d/%3d\n', episodeText, episodeIndex, MaxEpisodes);
                    data = [reshape(info.EpisodeReward, numAgents, 1), ...
                        reshape(stepCount, numAgents, 1), ...
                        reshape(info.AverageReward, numAgents, 1), ...
                        reshape(globalStepCount,  numAgents,1)];
                    varnames = {episodeRewardText, episodeStepsText, averageRewardText, stepCountText};
                    if any(this.HasCritic)
                        data = horzcat(data, reshape(info.EpisodeQ0, numAgents, 1) );
                        varnames{end+1} = episodeQ0Text;
                    end
                    statsTable = array2table(data);
                    statsTable.Properties.VariableNames = varnames;
                    [agentNames,~] = getAgentInfo(this);
                    statsTable.Properties.RowNames = agentNames;
                    disp(statsTable);
                else
                    % for single agent, display stats on each line
                    str = sprintf('%s: %3d/%3d | %s: %8.2f | %s: %4d | %s: %8.2f | %s: %4d', ...
                        episodeText,        episodeIndex,       MaxEpisodes, ...
                        episodeRewardText,  info.EpisodeReward, ...
                        episodeStepsText,   stepCount, ...
                        averageRewardText,  info.AverageReward, ...
                        stepCountText,      globalStepCount);
                    if any(this.HasCritic)
                        str = sprintf('%s | %s: %8.2f', str, episodeQ0Text, info.EpisodeQ0);
                    end
                    fprintf('%s\n', str);
                end
            end
        end
        function updateEpisodeManager(this,info)
            % push the training data onto the episode manager if
            % available
            episodeMgr = this.EpisodeMgr;
            if ~isempty(episodeMgr) && isvalid(episodeMgr) && ~this.Environment.TerminateSimulation
                stepEpisode(episodeMgr,info);
            end
        end
        function updateTrainingStats(this,info)
            % Keep track of statistics
            episodeIndex    = info.EpisodeCount;
            numAgents       = getNumAgents(this);
            for agentIndex = 1 : numAgents
                this.TrainingStats(agentIndex).TimeStamp     (episodeIndex) = string(duration(getDurationSinceReset(this.Watch),'Format','hh:mm:ss'));
                this.TrainingStats(agentIndex).EpisodeIndex   (episodeIndex) = episodeIndex;
                this.TrainingStats(agentIndex).EpisodeReward  (episodeIndex) = info.EpisodeReward(agentIndex);
                this.TrainingStats(agentIndex).EpisodeSteps   (episodeIndex) = info.EpisodeSteps(agentIndex);
                this.TrainingStats(agentIndex).AverageReward  (episodeIndex) = info.AverageReward(agentIndex);
                this.TrainingStats(agentIndex).TotalAgentSteps(episodeIndex) = info.GlobalStepCount(agentIndex);
                this.TrainingStats(agentIndex).AverageSteps   (episodeIndex) = info.AverageSteps(agentIndex);
                if any(this.HasCritic)
                    this.TrainingStats(agentIndex).EpisodeQ0  (episodeIndex) = info.EpisodeQ0(agentIndex);
                end
            end
        end
        function saveAgentToDisk(this,info)
            % save the agent to disk if the provided criteria has been met
            episodeIndex = info.EpisodeCount;
            if any(this.TrainingOptions.SaveAgentFunction(info))
                if getNumAgents(this) > 1
                    prefix = 'Agents';
                else
                    prefix = 'Agent';
                end
                SavedAgentFileName = fullfile(this.TrainingOptions.SaveAgentDirectory,[prefix, num2str(episodeIndex) '.mat']);
                saved_agent = this.Agents;
                stats = cleanupTrainingStats(this);
                savedAgentResultStruct = createSavedAgentResultStruct(this,stats);
                % make sure the saved agent is in sim mode
                wasMode = getStepMode(saved_agent);
                setStepMode(saved_agent,"sim");
                % g2419373 make sure to set MaxSteps = Inf before saving
                wasMaxSteps = [saved_agent.MaxSteps];
                [saved_agent.MaxSteps] = deal(Inf);
                try
                    save(SavedAgentFileName,'saved_agent', 'savedAgentResultStruct');
                    if ~isempty(this.EpisodeMgr) && isvalid(this.EpisodeMgr)
                        savedMessage = getString(message('rl:general:TrainingManagerSavedAgent',prefix,episodeIndex));
                        setActionMessage(this.EpisodeMgr,savedMessage);
                    end
                catch
                    % g1928023: We do not want to interrupt the training
                    % due to saving errors. Therefore a warning is thrown.
                    warning(message('rl:general:TrainingManagerUnableToSaveAgent',this.TrainingOptions.SaveAgentDirectory))
                end
                % change the mode and MaxSteps back
                for idx = 1:getNumAgents(this)
                    setStepMode(saved_agent(idx),wasMode(idx));
                    saved_agent(idx).MaxSteps = wasMaxSteps(idx);
                end
            end
        end
        
        function savedAgentResultStruct = createSavedAgentResultStruct(this,stats)
            numAgents = getNumAgents(this);
            agentName = createUniqueAgentNames(this);
            elapsedTime = char(getDurationSinceReset(this.Watch));
            for idx = numAgents:-1:1
                savedAgentInfoStruct(idx).EnvironmentName = string(getNameForEpisodeManager(this.Environment));
                savedAgentInfoStruct(idx).AgentName = agentName(idx);
                if isa(this.Environment,'AgentBlock')
                    savedAgentInfoStruct(idx).BlockPath = this.Environment.AgentBlock(idx);
                else
                    savedAgentInfoStruct(idx).BlockPath = [];
                end
                savedAgentInfoStruct(idx).TrainingOpts = this.TrainingOptions;
                savedAgentInfoStruct(idx).HasCritic = this.HasCritic;
                if ~isempty(this.Devices)
                    savedAgentInfoStruct(idx).HardwareResource = this.Devices(idx);
                else
                    savedAgentInfoStruct(idx).HardwareResource = [];
                end
                if ~isempty(this.LearnRates)
                    savedAgentInfoStruct(idx).LearningRate = this.LearnRates{idx};
                else
                    savedAgentInfoStruct(idx).LearningRate = [];
                end
                savedAgentInfoStruct(idx).TrainingStartTime = this.TrainingStartTime;
                savedAgentInfoStruct(idx).ElapsedTime = elapsedTime;
                savedAgentInfoStruct(idx).StopTrainingCriteria = this.TrainingStoppedReason(idx);
                savedAgentInfoStruct(idx).StopTrainingValue = this.TrainingStoppedValue(idx);
            end
            % saved agent result
            savedAgentResultStruct = struct(...
                'TrainingStats', stats,...
                'Information',   savedAgentInfoStruct);
        end
        
        function stopFlag = checkStopTraining(this,info)
            % stop training if the provided criteria has been met
            stopFlag = false;
            % Stop training (by stopping criteria or manually requested)
            % For multi-agents training resumes until all agents have met
            % their stopping criteria, or manually stopped.
            if all(this.TrainingOptions.StopTrainingFunction(info)) || ...
                    this.Environment.TerminateSimulation || ...
                    this.EpisodeCount >= this.TrainingOptions.MaxEpisodes || ...
                    all(this.TrainingStatus==0)
                stopFlag = true;
            end
        end
        function stopFlag = updateDisplaysFromTrainingInfo(this,info)
            % update the user visible components
            updateCommandLineDisplay(this,info);
            updateEpisodeManager(this,info);
            % update training stats
            updateTrainingStats(this,info);
            % save agent to disk if requested
            saveAgentToDisk(this,info);
            % stop training
            stopFlag = checkStopTraining(this,info);
        end
        function request2ManuallyTerminateCB(this,~,ed)
            % callback to manually terminate training
            terminateSimulation(this.Environment,ed.AffectedObject.RequestToStopTraining);
        end
        function [agentNames,agentTypes] = getAgentInfo(this)
            % return agent block names and types from the environment.
            numAgents = getNumAgents(this);
            agentNames = "";
            for idx = numAgents:-1:1
                if isa(this.Environment,'rl.env.SimulinkEnvWithAgent')
                    blkPaths = this.Environment.AgentBlock;
                    blkNames(idx) = string(get_param(blkPaths(idx),'Name'));
                    if numel(find(blkNames(idx)==blkNames)) > 1
                        % if there is another block with the same name (e.g.
                        % under a different subsystem) display full path
                        agentNames(idx) = blkPaths(idx);
                    else
                        agentNames(idx) = blkNames(idx);
                    end
                end
                agentTypes(idx) = string(regexprep(class(this.Agents(idx)),'\w*\.',''));
            end
        end
        function nameList = createUniqueAgentNames(this)
            numAgents = getNumAgents(this);
            nameList = "";
            for idx = numAgents:-1:1
                nameList(idx) = string(regexprep(class(this.Agents(idx)),'\w*\.',''));
            end
            nameList = matlab.lang.makeUniqueStrings(nameList);
        end
    end
    methods(Hidden)
        function mgr = getEpisodeManager(this)
            mgr = this.EpisodeMgr;
        end
        function qeSaveAgentToDisk(this,info)
            saveAgentToDisk(this,info);
        end
    end
end

%% local utility functions
function str = iDateTimeAsStringUsingDefaultLocalFormat(dt)
defaultFormat = datetime().Format;
dt.Format = defaultFormat;
str = char(dt);
end