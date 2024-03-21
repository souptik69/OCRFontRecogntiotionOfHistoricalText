import torch


class NoisyTeacherForcing():
    def __init__(self, A_size, noise_prob=0.):
        self.noise_prob = torch.Tensor([noise_prob])
        self.A_size = A_size

    def __call__(self, x):
        noise = torch.randint(low=0, high=self.A_size, size=x.shape)
        prob = torch.rand(size=x.shape)
        prob[:,0] = 1
        if x.is_cuda:
            noise = noise.cuda()
            prob = prob.cuda()
            self.noise_prob = self.noise_prob.cuda()
        return torch.where(prob>self.noise_prob,x,noise)

 
    
####  Font Detection ####
class NoisyTeacherFont():
    def __init__(self, character_vocab_size, font_class_count, noise_prob=0.):
        self.character_vocab_size = character_vocab_size
        self.font_class_count = font_class_count
        self.noise_prob = torch.Tensor([noise_prob])

    def __call__(self, x):
        batch_size, seq_len, _ = x.shape
        noise = torch.randint(low=0, high=self.character_vocab_size, size=(batch_size, seq_len, 1))
        font_noise = torch.randint(low=0, high=self.font_class_count, size=(batch_size, seq_len, 1))
        prob = torch.rand(size=x.shape)
        prob[:,0] = 1
        noisy_x = torch.cat((noise, font_noise), dim=2)
        if x.is_cuda:
            noise = noise.cuda()
            font_noise = font_noise.cuda()
            prob = prob.cuda()
            self.noise_prob = self.noise_prob.cuda()
            noisy_x= noisy_x.cuda()
        return torch.where(prob > self.noise_prob, x, noisy_x)
####  Font Detection  ####


if __name__ == "__main__":
    # NTF = NoisyTeacherForcing(A_size=89, noise_prob=0.8)
    # x = torch.LongTensor([[0,5,6,7,78,5,6,7,2,3,3,3,3,3,3,3], [0,5,6,7,78,5,6,7,2,3,3,3,3,3,3,3]]).cuda()
    # print(x-NTF(x))
    character_vocab_size = 89  # Example character vocabulary size
    font_class_count = 11     # Example number of font classes
    NTF1 = NoisyTeacherFont(character_vocab_size, font_class_count, noise_prob=0.8)
    x1 = torch.cat((torch.LongTensor([[0, 5], [1, 10]]), torch.LongTensor([[2, 6], [3, 9]])))
    print(x1 - NTF1(x1))