# Week 00

## progress

### day - 1

- implemented micrograd [grad.py]
- refreshed calc knowledge: derivatives, chain rule (it's everywhere) 
- operator overloading 
- traced graphs / used topo-sort (thought i was smart by implementing dfs first haha)  

### day - 2

- implemented MLP from scratch [mlp.py]
- applied gradient descent to minimize quadratic loss

### day - 3

- mastered micrograd & MLP (implement without looking at any references)
- added trigonometric functions to the engine
- completed 1st part of micrograd google colab challenge by karpathy [micrograd_exercises.ipynb]

### day - 4

- fully analyzed the whole micrograd repo
- completed google colab exercises for micrograd
- learned softmax, NLL loss, ReLU, Hinge loss, L2 penalty

### day - 5

vibe-coded an interactive 3d visualizer on top of micrograd using threejs. descent is not perfect yet. try it here: https://vivid-descent.vercel.app

### day - 6

< started makemore series, implemented bigram using manual nromalization
< trained a resnet-18 model based on PlantVillage dataset to detect diseases for hackathon https://tryapollo.vercel.app

### day - 7

< built bigram using neural nets, implemented nll loss & softmax
< practiced more pytorch

### day - 8

< wrote character prediction mlp
< started bengio et al 2003

### day - 9

# skipped a day, participated @ startup comp

< back to part 1 of makemore to complete exercises
< implemented trigram character prediction
< compared training (0.8) vs dev(0.1) vs test(0.1) 

some cool names i got:
hemma
jereva
peoff
vuvion
pochu
cuby
bugly
natte
lushi
miliamo


### i won the hackathon!
Excuse

Had to skip 2 days of "Chasing @karpathy" challenge in preparation for AI500 hackathon. Everything paid off. We presented our plant disease detection CNN on the competition and somehow won! @HusanIsamiddin cooked with the pitch, @umarHQ & @AbrorbekNemat0v with research & pitchdeck. 
Grateful for everything & everyone.

Back to work now.

### day - 10

< debugged day 9's trigram, removed repeated one hot enc assignments
< finished all the exercises from makemore 1, compared different L2 regularization lambdas λ

### day - 11~13

- read through Bengio et al. 2003, first ever paper i completed. felt like reading a sci-fi novel from the 80s.
- completed makemore 2 video

learned:
< about prev. approaches to statistical language models (e.g class-based, back-off, interpolated n-grams or LSI) and how can they be used as mixture
< how embeddings significantly help with much longer context lengths and take into account the "similarity" between words
< linear projection shortcut or direct input-to-output connections as paper refers (y=U * tanh(H*x + d) + W*x + b)
< data- & parameter-parallel processing used in the paper
< about energy minimization by incorporating embeddings for the output words

### hiatus

I am pausing the challenge for the next two weeks. Apparently, there are some colleges I need to apply to. Let's see how the sprint goes.

p.s. i will be polishing some of my projects in my free time

## projects
    first month:
    - https://tryapollo.vercel.app - plant disease detection model [pytorch, cnn]
    - https://vivid-descent.vercel.app - neural nets visualizer [threejs]
    - https://github.com/bilalsea2/cv-gallery - computer vision powered image gallery [mediapipe, nextjs]
