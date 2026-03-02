import cusrl
from cusrl_test import create_dummy_env, run_environment_evaluation_loop


def test_environment_with_observation_only():
    environment = create_dummy_env()
    agent = cusrl.preset.fastsac.AgentFactory(replay_batch_size=32).from_environment(environment)
    run_environment_evaluation_loop(environment, agent)


def test_environment_with_observation_and_state():
    environment = create_dummy_env(with_state=True)
    agent = cusrl.preset.fastsac.AgentFactory(replay_batch_size=32).from_environment(environment)
    run_environment_evaluation_loop(environment, agent)
