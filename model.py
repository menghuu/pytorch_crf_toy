#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2020 m <m@meng.hu>
#
# Distributed under terms of the MIT license.

"""
一个几乎是完全抄袭https://github.com/kmkurn/pytorch-crf的带注释版的不带验证crf

只是为了增加一些注释用来查看crf的原理
"""
import torch
from torch import nn
from torch import Tensor
from typing import List, Optional


class CRF(nn.Module):
    def __init__(self, num_tags):
        super(CRF, self).__init__()

        self.num_tags = num_tags
        self.start_transitions = nn.Parameter(torch.empty(num_tags))
        self.end_transitions = nn.Parameter(torch.empty(num_tags))
        self.transitions = nn.Parameter(torch.empty(num_tags, num_tags))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize the transition parameters.
        The parameters will be initialized randomly from a uniform distribution
        between -0.1 and 0.1.
        """
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)
        nn.init.uniform_(self.transitions, -0.1, 0.1)

    def forward(
            self,
            emissions: torch.Tensor,
            tags: torch.LongTensor,
            mask: Optional[torch.ByteTensor] = None,
            reduction: str = 'sum',
    ) -> torch.Tensor:
        if mask is None:
            mask = torch.ones_like(tags, dtype=torch.uint8)

        # shape: (batch_size,)
        numerator = self._compute_score(emissions, tags, mask)
        # shape: (batch_size,)
        denominator = self._compute_normalizer(emissions, mask)
        # shape: (batch_size,)
        llh = numerator - denominator

        return llh.sum()
    def decode(self, emissions: torch.Tensor,
               mask: Optional[torch.ByteTensor] = None) -> List[List[int]]:
        """Find the most likely tag sequence using Viterbi algorithm.
        Args:
            emissions (`~torch.Tensor`): Emission score tensor of size
                ``(seq_length, batch_size, num_tags)`` if ``batch_first`` is ``False``,
                ``(batch_size, seq_length, num_tags)`` otherwise.
            mask (`~torch.ByteTensor`): Mask tensor of size ``(seq_length, batch_size)``
                if ``batch_first`` is ``False``, ``(batch_size, seq_length)`` otherwise.
        Returns:
            List of list containing the best tag sequence for each batch.
        """
        if mask is None:
            mask = emissions.new_ones(emissions.shape[:2], dtype=torch.uint8)

        return self._viterbi_decode(emissions, mask)

    def _compute_score(self, emission: Tensor, tags: Tensor, mask: Tensor):
        # emission: [batch_size, seq_length, num_tags]
        # tags: [batch_size, seq_length]
        # mask: [batch_size, seq_length]
        # ~~对于输入来说，必须第一个位置是开头的标志，并且需要一个位置是结尾的标志~~
        # 但是如果第一个位置保证是<start>的话，以及结尾是<end>的tag的话，那么实际上这个 self.start_transitions 实际上只会有一个位置会被训练到，其他的都不会被训练到
        # 那么这个训练到的值有什么意义？ ~~似乎只相当于训练了一个偏执，而且还是一个全局的偏执~~
        # 这里的compute_score 只用到了一个位置，但是下面的compute_normalizer用到了其他的部分的内容，所以会将这个位置的值比例尽量算的大

        batch_size, seq_length = tags.shape

        score = self.start_transitions[tags[:, 0]]
        score += emission[torch.arange(batch_size), 0, tags[:, 0]]

        for i in range(1, seq_length):
            score += self.transitions[tags[:, i-1], tags[:, i]] * mask[:, i]
            score += emission[torch.arange(batch_size), i, tags[:, i]] * mask[:, i]

        seq_ends = mask.long().sum(dim=1) - 1
        # shape: (batch_size,)
        last_tags = tags[torch.arange(batch_size), seq_ends]
        # shape: (batch_size,)
        score += self.end_transitions[last_tags]

        # 其实从这里也能看出来，如果将数据中添加了<start>和<end>的效果是一样的，都是在计算开头和结尾中 transitions 的一个比例
        # start_transitions 表示一个我们看不到的开头转移到我们真实的开头的一个转移概率
        # 同理， end_transitions 表示我们真实的结尾转移到一个我们看不到的结尾的一个转移概率
        # 如果我们加了start_transitions 那么 end_transitions 也应该加上
        # 其实我们不在真实的数据开头加上<start>， 不在结尾加上<end>也行

        return score

    def _compute_normalizer(self, emissions, mask):
        # emissions: [batch_size, seq_length, num_tags]
        # mask: [batch_size, seq_length]

        seq_length = emissions.shape[1]
        num_tags = emissions.shape[-1]

        # [batch_size, num_tags]
        scores = self.start_transitions.unsqueeze(0) + emissions[:, 0, :]

        for i in range(1, seq_length):
            # [batch_size, num_tags, 1]
            expand_scores = scores.unsqueeze(2)

            # [batch_size, 1, num_tags]
            expand_emission = emissions[:, i, :].unsqueeze(1)

            # [batch_size, num_tags, num_tags]
            next_scores = expand_scores + expand_emission + self.transitions.unsqueeze(0)

            # [batch_size,           num_tags]
            next_scores = torch.logsumexp(next_scores, 1)

            scores = torch.where(mask[:, i].unsqueeze(1).bool(), next_scores, scores)

        scores += self.end_transitions

        # [batch_size]
        score = torch.logsumexp(scores, 1)

        return score

    def _viterbi_decode(self, emissions, mask):
        # 和计算normalizer的时候差不多，只是这个步骤中记录下

        batch_size = emissions.shape[0]
        seq_length = emissions.shape[1]
        history = []

        # [batch_size, num_tags]
        scores = self.start_transitions.unsqueeze(0) + emissions[:, 0, :]

        for i in range(1, seq_length):
            # [batch_size, num_tags, 1]
            expand_scores = scores.unsqueeze(2)

            # [batch_size, 1, num_tags]
            expand_emission = emissions[:, i, :].unsqueeze(1)

            # [batch_size, num_tags, num_tags]
            next_scores = expand_scores + expand_emission + self.transitions.unsqueeze(0)

            # 对应于1时刻的每个输出，0时刻哪个输入能使得输出是最大的
            # 对应于i时刻的每个输出，i-1时刻哪个输入能使得输出是最大的
            next_scores, indices = next_scores.max(dim=1)

            # 因为history是从0开始放进来的，不是从这个i的1开始的，所以他更像是说
            # k位置的哪个影响到k+1时刻下的 [0, nums_tag) 的影响能力大小
            history.append(indices)

            scores = torch.where(mask[:, i].unsqueeze(1).bool(), next_scores, scores)

        # 最后一个位置是虚拟位置，不存在的位置，因为它已经超过了mask的范围了
        # [batch_size, num_tags]
        scores += self.end_transitions.unsqueeze(0)

        best_tags_list = []
        seq_ends = mask.long().sum(dim=1) - 1

        # [batch_size]
        for batch_id in range(batch_size):
            tags = []
            index = scores.argmax(dim=1)[batch_id].item()
            tags.append(index)
            each_seq_length = seq_ends[batch_id]
            for h in history[:each_seq_length][::-1]:
                index = h[batch_id][index].item()
                tags.append(index)
            best_tags_list.append(tags[::-1])
        return best_tags_list


if __name__ == '__main__':
    from torchcrf import CRF as OriginalCRC
    import random
    nums_tags = 10
    batch_size = 8
    seq_length = 16
    original_crf = OriginalCRC(nums_tags, batch_first=True)


    crf = CRF(nums_tags)

    crf.start_transitions.data = original_crf.start_transitions.data
    crf.end_transitions.data = original_crf.end_transitions.data
    crf.transitions.data = original_crf.transitions.data


    emissions = torch.randn((batch_size, seq_length, nums_tags)).float()
    tags = torch.randint(nums_tags, (batch_size, seq_length))
    mask = torch.ones_like(tags, dtype=torch.bool)

    for i in range(len(mask)):
        j = random.randint(5, seq_length - 5)
        mask[i][j:] = False
    mask = mask.bool()
    print(mask)



    original_score = original_crf.forward(emissions, tags, mask)
    score = crf.forward(emissions, tags, mask)

    print(f'original score: {original_score}, score: {score}')

    original_tags_list = original_crf.decode(emissions, mask)
    tags_list = crf.decode(emissions, mask)

    print(f'original tags list: {original_tags_list}')
    print(f' tags list:         {tags_list}')



