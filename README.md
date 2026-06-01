# Pixel Moon Lander

這是一個簡化版的登月小艇 (Moon Lander) 環境，專為強化學習 (Reinforcement Learning, RL) 中的卷積神經網路 (CNN) 訓練所設計。
遊戲邏輯與 http://moonlander.seb.ly/ 相似，但移除了複雜的背景與材質，僅保留最基本的幾何形狀與黑白灰階像素，非常適合 DQN (Deep Q-Network) 等演算法。

## 特色
* **簡化的像素畫面**：環境輸出為 `(84, 84, 1)` 的灰階 NumPy Array，這是深度強化學習 (如 Atari 遊戲) 的標準輸入尺寸。
* **物理模擬**：包含重力下墜、主引擎推力以及左右旋轉的慣性計算。
* **Farama Gymnasium 標準介面**：實作了標準的 `reset()` 和 `step()` 方法，可直接與 Stable Baselines 3, Ray RLlib 等機器學習庫整合。

## 檔案說明
* `lander_env.py` - 主要的遊戲物理引擎與渲染環境。繼承自 `gymnasium.Env`。
* `play_human.py` - 用來讓人類玩家試玩的腳本（透過 PyGame 將 84x84 放大為 420x420 以便觀看）。
* `train_cnn_example.py` - 展示如何取得像素矩陣 (Observation) 來餵給 CNN 模型。
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
   * 降落目標：在最底部的白色平台上「平穩且端正」地降落即可獲得 +100 的 Reward。

3. **供 CNN 訓練讀取使用**：
   ```bash
   python train_cnn_example.py
   ```
   這會展示如何用程式化的方式調用環境，並獲得 `(84, 84, 1)` 的像素狀態矩陣。
