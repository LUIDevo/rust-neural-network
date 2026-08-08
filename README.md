A repository for me to learn ML concepts in my favorite language: Rust.

I built a simple classification model from scratch among some other things in this repo, it currently is built to handle the Fashion-MNIST dataset. Currently the roadmap is for me to work my way through transformers and maybe CNN's after that.

Also, I will implement wgpu, and I learned that GPU programming is much more complicated than I thought. It involves staging and buffers and alot of other things that I didn't realise I needed. 

Dev Roadmap:
- [x] Fashion-MNIST (Dense baseline)
- [x] Document accuracy and speed
- [ ] Switch to flat vec + f32
- [ ] Conv2D + Pool + Flatten
- [ ] Fashion-MNIST v2 (CNN + CPU, compare to baseline)
- [ ] GPU (wgpu)
- [ ] Fashion-MNIST v2 (CNN + GPU, compare to baseline)
- [ ] Self-attention from scratch
- [ ] LayerNorm + Residual connections
- [ ] Multi-head attention + positional encoding
- [ ] Minimal transformer block

# Dense baseline:
```
epoch 8 step 0/469 loss 0.2547 acc 0.9062
epoch 8 step 100/469 loss 0.1891 acc 0.9219
epoch 8 step 200/469 loss 0.1913 acc 0.9141
epoch 8 step 300/469 loss 0.2480 acc 0.9141
epoch 8 step 400/469 loss 0.2135 acc 0.9219
epoch 8: loss 0.2330 acc 0.9116
epoch 9 step 0/469 loss 0.2428 acc 0.9062
epoch 9 step 100/469 loss 0.2411 acc 0.9141
epoch 9 step 200/469 loss 0.2206 acc 0.9141
epoch 9 step 300/469 loss 0.2277 acc 0.9062
epoch 9 step 400/469 loss 0.2917 acc 0.9062
epoch 9: loss 0.2252 acc 0.9150
Training time: 6491.704011498s
TEST loss 0.3263 acc 0.8857
Test time: 35.799948085s
```
