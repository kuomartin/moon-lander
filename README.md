# Radar Moon Lander

這是一個為強化學習 (Reinforcement Learning, RL) 所設計的登月小艇 (Moon Lander) 環境，專為 **多層感知器 (MLP)** 訓練所打造。
本專案有別於傳統提供像素畫面讓卷積神經網路 (CNN) 學習的版本，而是使用了類似「雷達」的射線觀測系統，讓代理程式 (Agent) 可以透過多條發射射線來感知地形與周遭環境。

## 特色
* **雷達觀測空間**：代理程式可發出多條射線（預設 20 條），測量與地形、邊界或降落平台之間的距離與類別。此外還提供座標、速度與剩餘燃料等資訊。
* **物理模擬**：包含重力下墜、主引擎推力以及左右旋轉的慣性計算。
* **Farama Gymnasium 標準介面**：實作了標準的 `reset()` 和 `step()` 方法，可無縫與 Stable Baselines 3 (SB3) 等強化學習函式庫整合。

## 檔案說明
* `lander_env.py` - 主要的遊戲物理引擎與渲染環境（繼承自 `gymnasium.Env`）。
* `play_human.py` - 提供人類玩家試玩的腳本，用來熟悉環境與物理操作。
* `train.py` - 使用 Stable Baselines 3 的 PPO 演算法進行模型訓練與測試的腳本。
* `requirements.txt` - 環境需求套件清單。

## 如何執行

1. **安裝相依套件**：
   ```bash
   pip install -r requirements.txt
   ```

2. **自行手動遊玩 (測試物理與畫面)**：
   ```bash
   python play_human.py
   ```
   **操作方式**：
   * `W` 或 `上方向鍵`：啟動主引擎 (抗拒重力往上飛)
   * `A` 或 `左方向鍵`：逆時針旋轉
   * `D` 或 `右方向鍵`：順時針旋轉
   * 降落目標：在最底部的白色平台上「平穩且端正」地降落即可獲得大幅度的獎勵。

3. **訓練強化學習模型**：
   ```bash
   python train.py
   ```
   * 腳本預設會訓練 `200,000` 步，並把模型儲存至 `ppo_radar_moon_lander.zip`，訓練日誌會寫入 Tensorboard 中。
   * 可加上 `--resume` 接續之前的模型訓練。

4. **測試已訓練好的模型**：
   ```bash
   python train.py --test
   ```
   這會載入 `ppo_radar_moon_lander.zip` 模型，並開啟視覺化視窗展示 Agent 學習到的登月策略。
