#!/usr/bin/env python
import torch
import torch.nn.functional as F

def scaled_dot_product_attention(query,key,value,mask=None):
  #param query:查询张量，形状[batch_size,num_heads,seq_len_q,dim]
  #各维度代表批处理大小，注意力头数，查询序列的长度，每个元素特征维度
  #param key:键张量
  #param value:值张量
  #param mask:掩码张量
  scores = torch.matmul(query,key.transpose(-2,-1))/(query.size(-1)**0.5)
  if mask is not None:
    scores = scores.masked_fill(mask == 0,float('-inf'))
  #按最后一维做softmax，得到注意力权重：
  attention_weights = F.softmax(scores,dim = -1)
  #权重与value相乘，得到注意力输出
  output = torch.matmul(attention_weights,value)
  return output,attention_weights

#模拟输入数据
batch_size = 3
seq_len = 3
dim = 4
num_heads = 1
query = torch.randn(batch_size,num_heads,seq_len,dim)
key = torch.randn(batch_size,num_heads,seq_len,dim)
value = torch.randn(batch_size,num_heads,seq_len,dim)

output_self,attn_weights_self = scaled_dot_product_attention(query,key,value)
print("自注意力输出形状",output_self.shape)
print("自注意力权重形状",attn_weights_self.shape)

#模型交叉注意力机制，假设query来自一个序列，key和value来自一个序列
batch_size = 3
seq_len = 3
dim = 4
num_heads = 1
seq_len_kv = 4
query_cross = torch.randn(batch_size,num_heads,seq_len,dim)
key_cross = torch.randn(batch_size,num_heads,seq_len_kv,dim)
value_cross = torch.randn(batch_size,num_heads,seq_len_kv,dim)

output_cross,attn_weights_cross = scaled_dot_product_attention(query_cross,key_cross,value_cross)
print("自注意力输出形状",output_cross.shape)
print("自注意力权重形状",attn_weights_cross.shape)


