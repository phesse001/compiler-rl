from compiler_gym.leaderboard.llvm_instcount import eval_llvm_instcount_policy
from compiler_gym.envs import LlvmEnv
from cg_dqn import train

def train_and_run(env: LlvmEnv) -> None:
    env.observation_space = "Autophase"
    train(env)

if __name__ == "__main__":
    eval_llvm_instcount_policy(train_and_run)
