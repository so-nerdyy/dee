# Hacker News post (draft for Nivin to post from his own account)

Post this after the arXiv paper is public. HN titles must be 80 characters or less, and
"Show HN" is for things people can try, which the repo is.

## Title (pick one)

1. `Show HN: I pre-registered my MoE offloading benchmarks, and most predictions failed` (79)
2. `Show HN: DEE – running DeepSeek-V4-Flash from SSD, with exact, byte-checked outputs` (80)

Recommendation: option 1. Honest negative results do well on HN, and it's the paper's real point.

## URL

Link the arXiv abstract page (`https://arxiv.org/abs/XXXX.XXXXX`). Put the GitHub link in the
first comment.

## First comment (post it yourself right after submitting)

> Hi HN, I'm Nivin, a student, and this is a project I've been working on since July.
>
> DEE runs DeepSeek-V4-Flash (a mixture-of-experts model whose experts take 146 GiB) on a
> single 24 GB GPU by keeping the experts on an NVMe SSD and a RAM cache and loading them as
> the router asks for them. That idea isn't new, and other systems are faster than mine. What I
> tried to do differently is make the evaluation able to fail:
>
> - **Exactness.** Every run records the token ids and a hash-chained log of every expert the
>   router picked in every layer. A change to caching or scheduling only counts if those are
>   byte-identical to the reference. "The text looks about right" doesn't count.
> - **Predictions frozen in git before any cloud run.** Speed, cache hit rate, disk reads per
>   token and cost, with ranges and kill thresholds. Later results get appended, nothing gets
>   edited.
>
> What happened (about $14 of Modal GPU time over 7 runs, 6 GPU setups):
>
> - Exactness held on every GPU. A Modal 2xT4 run in November reproduced, token for token, an
>   answer I sealed on Kaggle's 2xT4 in August.
> - Different GPU architectures give different (but self-consistent) tokens. I think it's
>   floating-point rounding flipping a near-tie in the router, but I haven't proven it.
> - 9 of my 12 testable speed predictions were killed: reality was 2-6x slower. The
>   bottleneck turned out to be copying expert bytes into pinned memory on the CPU, which my
>   model assumed was free.
> - Prefetching experts with a predictor didn't pay off: the predicted hits mostly weren't
>   used. A simulator bug had made it look good.
> - Swapping LRU for a "next-use" eviction rule fixed a cache that had a literally 0% hit
>   rate at batch 8, and gave 1.30x speedup against a predicted 1.27x.
> - Best speed: 6.3 tokens/s aggregate at batch 8 on an RTX PRO 6000; about 0.75 tok/s single
>   stream on an L4.
>
> To be clear about what this is *not*: it's not a cheaper way to run a model than an API. On
> rented GPUs it works out to roughly $0.19-0.42 per 1,000 tokens, which is roughly 140 to
> 2,300 times what API providers charge for this same model ($0.18-1.32 per million output
> tokens on OpenRouter, Nov 20 2026), because they keep the whole model in memory across
> many GPUs and batch hundreds of users together. Offloading is for running a model that doesn't
> fit on the GPU you have (locally, privately, offline), and this project is mostly about
> checking that kind of system honestly.
>
> Code, the frozen predictions, every scorecard and the scorer: https://github.com/so-nerdyy/dee
>
> I used AI coding assistants a lot for this; the design decisions and the evaluation rules are
> mine, and I'm happy to answer questions about any of it. I'd especially like to hear from
> anyone who can explain the cross-GPU token differences.

## Tips

- Don't ask friends to upvote; HN detects voting rings and it can bury the post.
- Answer critical comments directly and calmly; the "here's what failed" framing helps.
- If it doesn't take off, HN allows a repost after a while; you can also email
  hn@ycombinator.com to ask about the second-chance pool.
