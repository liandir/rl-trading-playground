import torch


def train_on_historical(
        agent, env, data,
        n_episodes,
        batch_size=32,
        update_interval=100,
        n_updates=8,
        max_steps=2000,
        warm_up=500,
        store=1,
        actor_lr=1e-4,
        critic_lr=1e-4,
        actor_m=0.0,
        critic_m=0.0,
        optim="AdamW"
    ):
    """Train the agent in the given environment."""
    agent.update_optimizers(actor_lr, critic_lr, optim=optim, a_m=actor_m, q_m=critic_m)

    if store:
        total_reward = []
        total_info = []
        total_loss = []

    for episode in range(1, n_episodes+1):
        start = torch.randint(len(data)-max_steps, size=[1]).item()
        state = env.reset(data[start]).to_tensor()

        if store:
            episode_reward = []
            episode_info = []
            episode_loss = []

        for i in range(max_steps):
            action = agent.act(state, explore=True) if i > warm_up else None
            
            next_state, reward, done, info = env.step(action, data[start+i])
            next_state = next_state.to_tensor()

            episode_reward.append(reward)
            episode_info.append(info)

            if i > warm_up:
                agent.buffer.store(
                    state.detach(),
                    action.detach(),
                    next_state.detach(),
                    torch.tensor(reward).to(env.dtype),
                    torch.tensor(done).to(env.dtype)
                )
        
                if (i+1) % update_interval == 0 and len(agent.buffer) >= batch_size:
                    loss_dicts = []
                    for _ in range(n_updates):
                        loss_dicts.append(agent.update(batch_size))
                    
                    loss_dict = {
                        key: sum([item[key] for item in loss_dicts]) / len(loss_dicts)
                    for key in loss_dicts[0]}

                    episode_loss.append(loss_dict)

                    msg = f"episode {episode} [{100*i/max_steps:.1f}%] - reward: {reward:.5f} - portfolio: {info['V']:.2f}€"
                    for key, val in loss_dict.items():
                        msg += f" - {key}: {val:.5f}"
                    print(msg, end="\r")

            if done or (i+1) >= max_steps:
                break

            state = next_state

        if store:
            total_loss.append(episode_loss)
            total_reward.append(episode_reward)
            total_info.append(episode_info)

        msg = f"episode {episode} [{100*i/max_steps:.1f}%] - total reward: {sum(episode_reward):.5f} - portfolio: {info['V']:.2f}€"
        if len(episode_loss) > 0:
            for key in episode_loss[0].keys():
                msg += f" - {key}: {sum([loss_dict[key] for loss_dict in episode_loss]) / len(episode_loss):.5f}"
        print(msg)
    
    if store:
        return (
            total_loss,
            total_reward,
            total_info,
        )
    
    return [], [], [], {}
