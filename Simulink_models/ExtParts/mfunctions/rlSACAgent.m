classdef rlSACAgent < rl.agent.AbstractAgentMemoryTarget
    % rlSACAgent: Implements Soft Actor-Critic agent
    % adaptive entropy weight
    % REVISIT: support RNN

    % Copyright 2020 The MathWorks Inc.

    properties (Access = private)
        % REVISIT: Implement resetImple when you add new property

        % Counter determines when to update the policy
        PolicyUpdateCounter

        % Counter determines when to update the critic
        CriticUpdateCounter
    end

    properties (SetAccess = private)
        % Critic function approximator
        Critic

        % Target critic function approximator
        TargetCritic

        % Actor function approximator
        Actor

        % Target Entropy
        TargetEntropy

        % Indicator determining tunable Entropy Weight
        LearnableEntropy

        % Weight for Entropy
        EntropyWeight
        
        % Optimizer for entropy weight
        EntropyOptimizer
    end

    methods
        function this = rlSACAgent(Actor, Critic, Option)
            % Constructor
            this = this@rl.agent.AbstractAgentMemoryTarget(Actor.ObservationInfo,Actor.ActionInfo);

            % extract observation and action info
            this.ActionInfo = Actor.ActionInfo;
            this.ObservationInfo = Actor.ObservationInfo;

            % REVISIT: Not support multiple action channels due to DLT
            % limitation (a loss layer cannot take multiple inputs)
            % multi actions might work with dlnetwork
            if numel(this.ActionInfo) > 1
                error(message('rl:agent:errSACMultiActionChannel'))
            end

            % set agent option
            this.AgentOptions = Option;

            % set representations
            setActor(this,Actor);
            setCritic(this,Critic);
            setEntropy(this, this.AgentOptions);
            this.HasCritic = true;
        end

        %==================================================================
        % Get/set
        %==================================================================

        function this = setCritic(this,Critic)
            % setCritic: Set the critic of the reinforcement learning agent
            % using the specified representation, CRITIC, which must be
            % consistent with the observations and actions of the agent.
            %
            %   AGENT = setCritic(AGENT,CRITIC)

            % validate critic is a single output Q representation
            validateattributes(Critic, {'rl.representation.rlQValueRepresentation'}, {'vector', 'nonempty'}, '', 'Critic');

            if numel(Critic) > 2
                error(message('rl:agent:errSACNumCriticGt2'))
            end

            for ct = 1:numel(Critic)
                % validate critics are single output Q representations
                if strcmpi(getQType(Critic(ct)),'multiOutput')
                    error(message('rl:agent:errSACMultiQ'))
                end

                % check if actor and critic created from same data specs and if
                % both stateless or both have state
                rl.agent.AbstractAgent.validateActorCriticInfo(Critic(ct),this.Actor)

                % validate against agent options
                rl.agent.AbstractAgentMemoryTarget.validateOptionRepCompatibility(this.AgentOptions, Critic(ct));

                % validate action and observation infos are same
                checkAgentRepDataSpecCompatibility(this, Critic(ct));
                
                Critic(ct) = accelerateGradient(Critic(ct),true);
            end

            % All critics need to have different weights. Otherwise, you
            % don't get any benefits of using multiple Q networks.
            rl.agent.rlSACAgent.checkCriticsAreDifferent(Critic);

            % set critic and target   critic
            this.Critic = Critic;
            this.TargetCritic = Critic;

            % Reset agent. Call resetImpl
            reset(this);
        end

        function Critic = getCritic(this)
            % getCritic: Return the critic representations vector, for the
            % SAC agent, SACAGENT.
            %
            %   CRITIC = getCritic(SAC3AGENT)

            Critic = this.Critic;
        end

        function this = setActor(this, Actor)
            % setActor: Set the actor of the SAC agent using the specified
            % representation, ACTOR, which must be consistent with the
            % observations and actions of the agent.
            %
            %   AGENT = setActor(AGENT,ACTOR)

            % validate actor is a deterministic actor representation
            validateattributes(Actor, {'rl.representation.rlStochasticActorRepresentation'}, {'scalar', 'nonempty'}, '', 'Actor');

            % check if actor and critic(s) created from same data specs and
            % if both stateless or both have state
            Critics = getCritic(this);
            for ct = 1:numel(Critics)
                rl.agent.AbstractAgent.validateActorCriticInfo(Critics(ct),Actor);
            end

            % validate against agent options
            rl.agent.AbstractAgentMemoryTarget.validateOptionRepCompatibility(this.AgentOptions, Actor);

            % validate action and observation infos are same
            checkAgentRepDataSpecCompatibility(this, Actor)

            % set SaturationStrategy
            Actor = setSaturationStrategy(Actor,'tanh');
            
            % set actor network
            BoundType = getBoundTypeAndIndices(this.ActionInfo);
            if BoundType{1} == "AllBounded" || BoundType{1} == "AllUnBounded"
                % assume single action channel, accelerate only for all
                % bounded and unbounded case
                Actor = accelerateGradient(Actor,true);
            end
            this.Actor = Actor;

            reset(this)
        end

        function Actor = getActor(this)
            % getActor: Return the actor representation, ACTOR, for the
            % specified reinforcement learning agent.
            %
            %   ACTOR = getActor(AGENT)
            %

            Actor = this.Actor;
        end
    end

    %======================================================================
    % Implementation of abstract methods
    %======================================================================
    methods
        function Action = getActionWithExploration(this,Observation)
            % Given the current state of the system, return an action
            % REVISIT support multiple channel action

            % getActionImpl from the stochastic actor returns stochastic
            % actions. Hence, we don't need to add any expoloration noise
            % in this function.

            % getAction returns unbounded action
            [Action, State] = getAction(this.Actor, Observation);
            this.Actor = setState(this.Actor, State);

            % saturate the unbounded action
            Action = saturateAction(this.Actor, Action);
        end
    end

    methods (Access = protected)
        function options = setAgentOptionsImpl(this,options)
            rl.util.validateAgentOptionType(options,'SAC');
            rl.agent.AbstractAgentMemoryTarget.validateOptionRepCompatibility(options, this.Actor);
            rl.agent.AbstractAgentMemoryTarget.validateOptionRepCompatibility(options, this.Critic);
            setEntropy(this,options);
        end

        function [rep,argStruct] = generateProcessFunctions_(this,argStruct)
            rep = this.Actor;
            argStruct = rl.codegen.generateContinuousStochasticPolicySquashFcn(argStruct,this.ActionInfo,this.AgentOptions.UseDeterministicExploitation);
        end

        function Action = getActionImpl(this, Observation)
            % Returns saturated action given observation
            % REVIST: does not support multi-action channel

            % 1. Get the distribution parameters by evaluating the network
            % 2. Compute the action using the reparameterization trick and
            % Gaussian distribution with the parameters obtained from the
            % network. This action is not bounded.
            % 3. Saturate the action based on the saturation strategy (none
            % or tanh) and limits in the ActionInfo.

            if this.AgentOptions.UseDeterministicExploitation
                % getDeterministicAction returns mean
                [Action, State] = getDeterministicAction(this.Actor, Observation);
            else
                % getAction returns unbounded action
                [Action, State] = getAction(this.Actor, Observation);
            end
            this.Actor = setState(this.Actor, State);

            % saturate the unbounded action
            Action = saturateAction(this.Actor, Action);
        end

        % set/get tunable parameters
        function setLearnableParametersImpl(this,p)
            this.Actor = setLearnableParameters(this.Actor, p.Actor);
            for ct = 1:numel(this.Critic)
                this.Critic(ct) = setLearnableParameters(this.Critic(ct),p.Critic{ct});
            end

            this.EntropyWeight = p.EntropyWeight;
        end
        function p = getLearnableParametersImpl(this)
            p.Actor  = getLearnableParameters(this.Actor );
            for ct = 1:numel(this.Critic)
                p.Critic{ct} = getLearnableParameters(this.Critic(ct));
            end
            p.EntropyWeight = this.EntropyWeight;
        end

        function varargout = learn(this,exp)
            % learn from the current set of experiences where
            % exp = {state,action,reward,nextstate,isdone}
            % Return the noisy action.

            % store experiences
            appendExperience(this,exp);

            % Update networks if ExperienceBuffer has enough samples.
            % REVISIT: NumWarmStartSteps should be less than the number of
            % all training steps.

            if this.ExperienceBuffer.Length >= max(this.AgentOptions.NumWarmStartSteps, this.AgentOptions.MiniBatchSize)
                if hasState(this)
                    % We need to reset state of actor for training. We store the current
                    % state to restore the state after this training step.
                    CurrentActorState = getState(this.Actor);
                    this.Actor = resetState(this.Actor);
                end

                % Training step
                stepRepresentationForMultipleGraidentSteps(this);

                if hasState(this)
                    % after the training, recover the saved hidden state.
                    this.Actor  = setState(this.Actor, CurrentActorState);
                end
            end

            if nargout
                % compute action from the current policy
                % {state,action,reward,nextstate,isdone}
                varargout{1} = getActionWithExploration(this,exp{4});
            end
        end

        function resetImpl(this)
            % rebuild agent properties due to any potential changes in
            % options
            resetImpl@rl.agent.AbstractAgentMemoryTarget(this);
            
            % utils to keep track of update
            this.PolicyUpdateCounter = 1;
            this.CriticUpdateCounter = 1;
        end

        function Q0 = evaluateQ0Impl(this,Observation)
            % overload for agents that implement critics
            Action = getAction(this.Actor, Observation);
            Action = saturateAction(this.Actor, Action);
            for ct = 1:numel(this.Critic)
                Q0(ct) = getValue(this.Critic(ct), Observation ,Action); %#ok<AGROW>
            end
            Q0 = min(Q0);
            if isa(Q0,'dlarray')
                Q0 = extractdata(Q0);
            end
        end

        function trainingOptions = validateAgentTrainingCompatibilityImpl(this,trainingOptions)

            validateAgentTrainingCompatibilityImpl@rl.agent.AbstractAgentMemoryTarget(this,trainingOptions);

            % Validate SAC agent training options compatibility
            if ~strcmpi(trainingOptions.Parallelization,'none')
                dataToSend = trainingOptions.ParallelizationOptions.DataToSendFromWorkers;
                % SAC agent only support send experiences for parallel
                if ~strcmpi(dataToSend,'Experiences')
                    error(message('rl:general:errParallelSendGradNotSupport'));
                end
            end
        end

        function HasState = hasStateImpl(this)
            % whether use RNN
            HasState = hasState(this.Actor);
        end

        function resetStateImpl(this)
            % reset state of RNN policy, no-op for non-RNN
            this.Actor = resetState(this.Actor);
        end

        function that = copyElement(this)
            that = copyElement@rl.agent.AbstractAgent(this);
            that.ExperienceBuffer = copy(this.ExperienceBuffer);
        end

        %% Experience based parallel methods ==============================
        % Methods to support experience based parallel training
        % =================================================================

        % Return a policy for parallel training. SAC uses the action with
        % exploration for both training and testing.
        function policy = getParallelWorkerPolicyForTrainingImpl(this)
            % build the policy
            actor = this.Actor;
            % force the actor to use cpu as the workers will just be
            % executing 1 step predictions for simulation
            actor.Options.UseDevice = "cpu";
            policy = rl.policy.RepresentationPolicy(actor,getSampleTime(this));
        end

        % Get the actor learnable parameters
        function p = getParallelWorkerPolicyParametersImpl(this)
            % just return the parameters of the actor
            p = getLearnableParameters(this.Actor);
            % make sure to convert gpu params to cpu params
            p = dlupdate(@gather,p);
        end

        % Learn from experiences processed on the workers
        function learnFromSimulationWorkerExperiencesImpl(this,processedExperiences)
            % call learnFromExperiences which expects processed experiences
            learnFromExperiences(this,processedExperiences);
        end
    end

    %======================================================================
    % Implementation of hidden methods
    %======================================================================
    methods (Hidden)
        function preSettings = preTrain(this)
            preSettings = preTrain@rl.agent.AbstractAgent(this);
            preSettings.ValidateInputArgumentsForExperienceBuffer = ...
                this.ExperienceBuffer.DoValidate;
            this.ExperienceBuffer.DoValidate = false;
        end

        function postTrain(this,preSettings)
            postTrain@rl.agent.AbstractAgent(this,preSettings);
            this.ExperienceBuffer.DoValidate = ...
                preSettings.ValidateInputArgumentsForExperienceBuffer;
        end

        function appendExperience(this,experiences)
            % append experiences to buffer
            append(this.ExperienceBuffer,{experiences});
        end
    end

    %======================================================================
    % Step representation methods
    %======================================================================
    methods (Access = private)
        function this = setEntropy(this, options)
            if isempty(options.EntropyWeightOptions.TargetEntropy)
                % REVISIT: -|A| is a good hyperparameter when actions are
                % bounded [-1, 1] using tanh. In unbounded case, it may
                % need to use larger value than -|A| to encourage
                % exploration.
                this.TargetEntropy = -prod(this.ActionInfo.Dimension);
            else
                this.TargetEntropy = options.EntropyWeightOptions.TargetEntropy;
            end
            this.EntropyWeight = options.EntropyWeightOptions.EntropyWeight;
            this.LearnableEntropy = true;
            if options.EntropyWeightOptions.LearnRate == 0
                this.LearnableEntropy = false;
            end
            if this.LearnableEntropy
                this.EntropyOptimizer = rl.util.createSolverFactory(options.EntropyWeightOptions, this.EntropyWeight);
            end
        end

        function updateCriticTargetRepresentations(this)
            % Update the target critic representations

            for ct = 1:numel(this.Critic)
                this.TargetCritic(ct) = updateTarget(this,this.Critic(ct),this.TargetCritic(ct),...
                    this.AgentOptions.TargetSmoothFactor,...
                    this.AgentOptions.TargetUpdateFrequency);
            end
        end

        function stepRepresentationForMultipleGraidentSteps(this)
            UpdateCritic = mod(this.CriticUpdateCounter, this.AgentOptions.CriticUpdateFrequency) == 0;
            UpdatePolicy = mod(this.PolicyUpdateCounter, this.AgentOptions.PolicyUpdateFrequency) == 0;

            if UpdateCritic || UpdatePolicy
                % generate a minibatch: MiniBatchSize length of cell array with
                % {state,action,reward,nextstate,isdone} elements
                for epochInd = 1:this.AgentOptions.NumGradientStepsPerUpdate
                    if hasState(this)
                        [minibatch, maskIdx] = createSampledExperienceMiniBatchSequence(...
                            this.ExperienceBuffer,...
                            this.AgentOptions.MiniBatchSize,...
                            this.AgentOptions.SequenceLength);
                    else
                        minibatch = createSampledExperienceMiniBatch(...
                            this.ExperienceBuffer,...
                            this.AgentOptions.MiniBatchSize,...
                            this.AgentOptions.DiscountFactor,...
                            this.AgentOptions.NumStepsToLookAhead);
                        maskIdx = [];
                    end
                    if ~isempty(minibatch)
                        if UpdateCritic
                            trainCriticWithBatch(this,minibatch, maskIdx);
                            updateCriticTargetRepresentations(this);
                        end
                        if UpdatePolicy
                            trainActorWithBatch(this,minibatch, maskIdx);
                            trainEntropyWeight(this,minibatch,maskIdx);
                        end
                    end
                end
            end

            if UpdateCritic
                this.CriticUpdateCounter = 1;
            else
                this.CriticUpdateCounter = this.CriticUpdateCounter + 1;
            end

            if UpdatePolicy
                this.PolicyUpdateCounter = 1;
            else
                this.PolicyUpdateCounter = this.PolicyUpdateCounter + 1;
            end
        end

        function varargout = trainCriticWithBatch(this,miniBatch, maskIdx)
            % update the critics and actor against a minibatch set
            Observations     = miniBatch{1};
            Actions          = miniBatch{2};
            Rewards          = miniBatch{3};
            NextObservations = miniBatch{4};
            IsDones          = miniBatch{5};

            DistributionParams = evaluate(this.Actor, NextObservations);
            NextActions = getSampleAndNoise(this.Actor, DistributionParams);
            SaturatedNextActions = saturateAction(this.Actor, NextActions);

            % compute the next step expected Q value (bootstrapping)
            for ct = 1:numel(this.Critic)
                % reset representation state (RNN), no-op for non RNN
                this.Critic(ct) = resetState(this.Critic(ct));
                if ct < 2
                    TargetQ = getValue(this.TargetCritic(ct), NextObservations, SaturatedNextActions);
                else
                    TargetQ = min(TargetQ, getValue(this.TargetCritic(ct), NextObservations, SaturatedNextActions));
                end
            end

            Entropy = getEntropy(this.Actor, DistributionParams{1}, NextActions{1});
            LogDensities = -Entropy;
            
            TargetQ = TargetQ - this.EntropyWeight * LogDensities;
            
            DoneIdx = IsDones == 1;
            Gamma = this.AgentOptions.DiscountFactor;
            n = this.AgentOptions.NumStepsToLookAhead;

            % get target Q values we should expect the network to work
            % towards
            TargetQ(~DoneIdx) = Rewards(~DoneIdx) + (Gamma^n).*TargetQ(~DoneIdx);

            % for final step, just use the immediate reward, since there is
            % no more a next state
            TargetQ(DoneIdx) = Rewards(DoneIdx);

            % train the critic or get the gradients
            for ct = 1:numel(this.Critic)
                TargetQSingle = TargetQ;
                % dummification of q target from RNN patching
                if hasState(this)
                    % bypass of data is not patched
                    if ~all(maskIdx,'all')
                        QPrediction = getValue(this.Critic(ct),Observations,Actions);
                        TargetQSingle(~maskIdx) = QPrediction(~maskIdx);
                    end
                end
                CriticGradient(ct) = {gradient(this.Critic(ct),@rl.grad.rlmseLoss,...
                    [Observations, Actions], TargetQSingle)}; %#ok<AGROW>
            end

            if nargout
                s.Critic = CriticGradient;
            else
                for ct = 1:numel(this.Critic)
                    this.Critic(ct) = optimize(this.Critic(ct), CriticGradient{ct});
                end
            end
            if nargout
                varargout{1} = s;
            end
        end
        
        function trainActorWithBatch(this,miniBatch,maskIdx)
            % update the actor against a minibatch set
            % REVIST: Support multi-action channel
                        
            if hasState(this.Actor)
                % reset actor and critic state before learning
                this.Actor  = resetState(this.Actor);
                for ct = 1:numel(this.Critic)
                    this.Critic(ct) = resetState(this.Critic(ct));
                end
                % numExperience == number of non patched experience
                numExperience = sum(maskIdx,'all');
            else
                % numExperience == number of rewards
                numExperience = numel(miniBatch{3});
            end
            
            % compute actor gradient
            for ct = 1:numel(this.Critic)
                gradInput.Critic{ct} = getModel(this.Critic(ct));
                gradInput.CriticInputIndex{ct} = getInputIndex(this.Critic(ct));
            end
            gradInput.MaskIdx = maskIdx;
            gradInput.NumObs = numExperience;
            gradInput.SamplingStrategy = getSamplingStrategy(this.Actor);
            gradInput.EntropyWeight = this.EntropyWeight;
            gradInput.ActorInputIndex = getInputIndex(this.Actor);
            actorGradient = customGradient(this.Actor,@rl.grad.sac,miniBatch{1},gradInput);
            
            % update actor
            this.Actor = optimize(this.Actor, actorGradient);
        end
        
        function trainEntropyWeight(this, miniBatch,maskIdx)
            if this.LearnableEntropy
                Observations = miniBatch{1};
                DistributionParams = evaluate(this.Actor, Observations);
                Actions = getSampleAndNoise(this.Actor, DistributionParams);
                Entropy = getEntropy(this.Actor, DistributionParams{1}, Actions{1});
                LogDensities = -Entropy;
                if hasState(this.Actor)
                    % Size of the entropy should be the same as maskIdx
                    % 1 x BatchSize x SequenceLength (even if the action
                    % dimension is 2, the first dimension is 1)
                    % nan will be ignored when mean is computed.
                    LogDensities(~maskIdx) = nan;

                    % REVISIT when dlnetwork is used. omitnan is not
                    % supported.
                    EntropyGradient = - mean( LogDensities + this.TargetEntropy, 'all','omitnan');
                else
                    EntropyGradient = - mean( LogDensities + this.TargetEntropy, 'all');
                end

                EntropyGradient = max(EntropyGradient, -this.AgentOptions.EntropyWeightOptions.GradientThreshold);
                EntropyGradient = min(EntropyGradient, this.AgentOptions.EntropyWeightOptions.GradientThreshold);
                LogEntropyWeight = log(rl.internal.dataTransformation.boundAwayFromZero(this.EntropyWeight));
                [this.EntropyOptimizer, UpdatedLogEntropyWeight] = calculateUpdate(this.EntropyOptimizer, {LogEntropyWeight}, ...
                    {EntropyGradient}, this.AgentOptions.EntropyWeightOptions.LearnRate);
                LogEntropyWeight = UpdatedLogEntropyWeight{1};
                this.EntropyWeight = exp(LogEntropyWeight);
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%
                % Qinglei revises here:
                EntropyWeightList = evalin('base', 'EntropyWeightList');
                EntropyList = evalin('base', 'EntropyList');
                
                addEntropyToIndex = find(EntropyWeightList,1,'Last')+1;

                EntropyWeightList(addEntropyToIndex) = this.EntropyWeight;
                EntropyList(addEntropyToIndex) = mean(Entropy);
                assignin('base', 'EntropyWeightList', EntropyWeightList);
                assignin('base', 'EntropyList', EntropyList);
                % EntropyList = [EntropyList this.EntropyWeight]
            end
        end
    end

    methods (Access = private)
        function checkAgentRepDataSpecCompatibility(this, Rep)
            if ~isCompatible(this.ActionInfo, Rep.ActionInfo)
                error(message('rl:agent:errActionInfoAC'))
            end
            if ~isCompatible(this.ObservationInfo, Rep.ObservationInfo)
                error(message('rl:agent:errObservationInfoAC'))
            end
        end
    end
    
    methods (Hidden, Static)
        function checkCriticsAreDifferent(Critic)
            % Input: a vector of Critic networks
            % If Critics are the same, it errors out.
            % All critics need to have different weights. Otherwise, you
            % don't get any benefits of using multiple Q networks.
            if numel(Critic)>1
                CriticPairs = nchoosek([1:numel(Critic)],2);
                for pairIndex = 1:size(CriticPairs,1)
                    Ct1 = CriticPairs(pairIndex,1);
                    Ct2 = CriticPairs(pairIndex,2);

                    % Check whether two critics have the same weight
                    IsEqualWeight = isequal(getLearnableParameters(Critic(Ct1)), getLearnableParameters(Critic(Ct2)));
                    % Check whether two critics have the same architecture. Even
                    % if values of weights are the same, they may use different
                    % layers.
                    IsEqualStructure = isequal(Critic(Ct1), Critic(Ct2));

                    if IsEqualWeight && IsEqualStructure
                        error(message('rl:agent:errDoubleQSame'))
                    end
                end
            end
        end

        function validateActionInfo(ActionInfo)
            % REVISIT: move to superclass of SAC
            if ~isa(ActionInfo,'rl.util.RLDataSpec')
                error(message('rl:agent:errInvalidActionSpecClass'))
            end

            % SAC does not support discrete action data spec
            if rl.util.isaSpecType(ActionInfo, 'discrete')
                error(message('rl:agent:errSACContinuousActionSpec'))
            end

            % REVISIT: Not support multiple action channels due to DLT
            % limitation (a loss layer cannot take multiple inputs)
            % multi actions might work with dlnetwork
            if numel(ActionInfo) > 1
                error(message('rl:agent:errMultiActionChannelNotSupport'))
            end
        end
    end
    
    methods (Static)
        function obj = loadobj(s)
            if isstruct(s)
                obj = rl.agent.rlSACAgent(s.Actor,s.Critic,s.AgentOptions_);
                obj.TargetCritic = s.TargetCritic;
                obj.EntropyWeight = s.EntropyWeight;
                
                if obj.AgentOptions_.SaveExperienceBufferWithAgent
                    % only load the experience buffer if
                    % SaveExperienceBufferWithAgent is true
                    obj.ExperienceBuffer = s.ExperienceBuffer;
                end
            else
                obj = s;
            end
        end
    end
end
