import torch
import torch.nn.functional as F
# b = batch size
# s = sequence length
# m = hidden size
# l = batch size * sequence length
# e = number of experts
# c = expert capacity

# input has shape (b,s,m)
hidden_states = torch.tensor([[1,2,3,4],[5,6,7,8]], dtype=torch.float).reshape(2,4,1)
# reshaped hidden states has shape (l,m)
reshaped_hidden_states = hidden_states.reshape(-1,1)

# one host dispatch has shape (b,s,e). One could pre drop tokens by settings all values to 0 in the one-hot-encoding of a token (last dimension).
one_hot_dispatch = torch.tensor([[[1,0],[1,0],[0,1],[1,0]],[[0,1],[1,0],[0,1],[1,0]]], dtype=torch.float)
# reshaped input has shape (l,e)
reshaped_one_hot = one_hot_dispatch.reshape(-1,2)

expert_capacity = 4

# Has shape (l,e). Cumulative sum on each expert to compute the capacity index that each token has for its given index.
index_from_cumsum = (torch.cumsum(reshaped_one_hot, dim=0) - 1) * reshaped_one_hot
# Has shape (l). Given j=token_capacity_index[i], it means that token "i" has is at index "j" of its expert's capacity.
token_capacity_index = torch.sum(index_from_cumsum, dim=1).to(torch.int64)
# Here we need to handle the fact some experts might receive more tokens than the capacity. In this case, we drop the tokens.
drop_mask = torch.lt(token_capacity_index, expert_capacity)
token_capacity_index = token_capacity_index * drop_mask
# Contains the position of each token in its expert's capacity. Has shape (l,c). For example, given a=token_to_expert[i], then the non-zero entry in "a" is where token "i" would be in the tokens assigned to the expert.
token_to_expert = F.one_hot(token_capacity_index, num_classes=expert_capacity)
token_to_expert = token_to_expert * drop_mask.reshape(*token_capacity_index.shape, 1).expand(*token_to_expert.shape)
# Has shape "lec".
dispatch_mask = torch.einsum('le,lc->lec', reshaped_one_hot, token_to_expert)

# Has shape (e,c,m). Each entry dispatched_tokens[i] contains the tokens assigned to expert "i". An entry dispatched_tokens[i][j] contains the hidden states of some token assigned to expert "i".
dispatched_tokens = torch.einsum('lec,lm->ecm', dispatch_mask, reshaped_hidden_states)

print("num experts: 2")
print(f"expert capacity: {expert_capacity}")
print(f"hidden_states ({hidden_states.shape})")
print(hidden_states)
print(f"one_hot_dispatch ({one_hot_dispatch.shape})")
print(one_hot_dispatch)
print(f"dispatched_tokens ({dispatched_tokens.shape})")
print(dispatched_tokens)
print("notice that the token 8 did not get assigned to expert 0 as the capacity overflowed")
