import cv2


def render_learning(num_timesteps, env, model):
    for _ in range(num_timesteps):
        action, _ = model.predict(obs, deterministic=True)
        obs, _reward, done, _trunacted, _info = env.step(action)
        img = env.render()

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imshow("", img)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        if done:
            obs, _info = env.reset()
