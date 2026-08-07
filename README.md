A repository for me to learn ML concepts in my favorite language: Rust.

I built a simple classification model from scratch among some other things in this repo, it currently is built to handle the Fashion-MNIST dataset. Currently the roadmap is for me to work my way through transformers and maybe CNN's after that.

Also, I will implement wgpu, and I learned that GPU programming is much more complicated than I thought. It involves staging and buffers and alot of other things that I didn't realise I needed. 

Dev Roadmap:
- [x] Fashion-MNIST (Dense baseline)
- [ ] Document accuracy and speed
- [ ] Switch to flat vec + f32
- [ ] Conv2D + Pool + Flatten
- [ ] Fashion-MNIST v2 (CNN + CPU, compare to baseline)
- [ ] GPU (wgpu)
- [ ] Fashion-MNIST v2 (CNN + GPU, compare to baseline)
- [ ] Self-attention from scratch
- [ ] LayerNorm + Residual connections
- [ ] Multi-head attention + positional encoding
- [ ] Minimal transformer block
