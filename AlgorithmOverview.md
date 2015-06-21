# Introduction #

The idea for this algorithm is simple. The idea is to imagine that individuals pick a strategy profile (the percentage of the time that they will play each of their pure strategies). After playing it for a fixed amount of time they reevaluate their decision. Pure strategies that had an expected payoff lower than the average payoff experienced they will play less in the future. Pure strategies that did better than average they play more often. Now if you imagine that players can do these sorts of calculations in an instantaneous fashion, this logic can be used to setup each players adaptive rule as a system of differential equations. The solution to which, when it converges, will converge to a Nash equilibrium. One case when the system of differential equations will not converge is when there is only a single mixed strategy Nash equilibrium. In this case the system will cycle around the equilibrium, with the equilibrium at its center. That the equilibrium is at the center of the cycle can be used to find the equilibrium when the systems do not converge.

## The Math ##

Consider the congestion game shown below in normal form. There are 2 players, each of which has two routes they can take.

**Payoffs**

| | Road 1 | Road 2|
|:|:-------|:------|
| Road 1| {2,2}  | {1, 0.5}|
| Road 2| {0.5, 1}| {1,1} |

Let's assign some variables to the probability that each player chooses each road.

Let P1 be the probability that player 1 chooses road 1.
Let P2 be the probability that player 1 chooses road 2.
Let Q1 be the probability that player 2 chooses road 1.
Let Q2 be the probability that player 2 chooses road 2.

dP1/dt = P1((2\*Q1+1\*Q2) - (P1(2\*Q1+1\*Q2)+P2(0.5\*Q1+1\*Q2)))

dP2/dt = P2((0.5\*Q1+1\*Q2) - (P1(2\*Q1+1\*Q2)+P2(0.5\*Q1+1\*Q2)))

dQ1/dt = Q1((2\*P1+1\*P2) - (Q1(2\*P1+1\*P2)+Q2(P1\*0.5+P2\*1)))

dQ2/dt = Q1((P1\*0.5+P2\*1) - (Q1(2\*P1+1\*P2)+Q2(P1\*0.5+P2\*1)))


So the general idea is that the change is the probability of playing a strategy is:

(The current probability of playing the strategy) Times  ((the expected utility of playing the strategy) Minus (the average utility over all strategies)).

Because probabilities must sum to one, including P2 and Q2 are redundant. However, from a computational standpoint, worrying about doing the substitution Qn= (1-Sum(Qi to Q(n-1)) could be more trouble than including it.



# Computational strategies #
There are several potential approaches to trying to parralize the computation of this set of differential equations. The Single Instruction Multiple Data (SIMD) architecture of graphics cards makes algorithms that rely on vector operations easy to implement via CUDA. Also, algorithms that require a lot of branching and logical checks seem to be less likely to benefit from this type of parallel architecture.

## Idea #1 ##
Good data structures will make vector operations easier to code.

Data structure #1
Imagine that we write out the description of the game as a vector.

We could write out the congestion game from before as:

`{{{1,1},{2,2}},{{1,2},{1, 0.5}},{{2,1},{0.5, 1}}, {{2,2},{1,1}}} `

Where each element of the top level vector represents one of possible outcomes of the game. Within each top level element in another vector. For instance the first element is
{{1,1},{2,2}}
which has two sub vectors. The first of these sub-vectors denotes the strategies played be each of the players for that outcome of the game. In this case both players took the first road. The next of these sub-vectors denotes the payoffs to each players from this outcome of the game. I think this is a sensible way to represent the game, but am not sure about how much of a pain it would be to pass a data structure like this in CUDA to the graphics card.

Data structure #2
Suppose there are N-players each with M strategies. Lets create an NXM matrix where the element [i,j] denotes that probability that player i plays strategy j.

The key step in this algorithm is calculating the expected payoff from following each strategy for each player.

Inputs
  1. variable game: data structure #1 (a vector of length N\*M)
  1. variable guess: an initial guess for the Nash equilibrium (an NXM matrix described in data structure #2)

Outputs
  1. an updated guess for the Nash equilibrium, updated according to the differential equations described above.

## Detour into CUDA details ##

An important concept in CUDA is that when you write a function, you can call it across a large number of blocks with each block containing a fixed number of threads. Each thread performs the same operation on a different element of a vector. So here the vector we care about is described in the variable game. Each thread has access to several variables global variables that help in determining which element it is to operate on. The variables are:
  1. blockDim.x which gives the number of threads in each block
  1. blockIdx.x which tells which block the thread is in
  1. threadIdx.x which tells the thread what its ID number is within the block.

So lets say we want to calculate the addition of two large vectors using CUDA. Each vector is 1024 elements long. First we define a function:

```
_global_ void VecAddition(float* vec1, float * vec2, float * result)
{
   int idx = blockDim.x * blockIdx.x + threadIdx.x;
   results[idx]=vec1[idx]+vec2[idx]
}  
```

Now when we call this function we have to tell the graphics card how many threads per block and how many blocks we want to execute the function over.

```
int main()
{
   ...
   dim3 dimBlock (256);
   dim3 dimGrid(4);
   VecAddition<<<dimGrid, dimBlock>>>(vec1,vec2,result);


}
```

The operator <<< Num blocks, Num threads per block>>> lets the complier know that you want to call the function on the graphics card hardware with the specified block and thread configuration. There are, however, quite a few memory allocation issues that are being glossed over because they aren't central to the algorithm.

## Using CUDA to solve for Nash equilibrium ##

My first thought is to operate on the game vector. That is to assign a thread to each element of the game vector thus needing N\*M threads for the function call.

The first step of the algorithm is extract the payoffs for each of the N\*M game outcomes for a particular player. We start with player 1.

```
_global_ void ExpectedPay(GameType * game, float ** guess, float * pay1,int N, int M)
{
  int i=0;
  int j=0;
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
   for(i=0; i<N; i++)
   {
      for(j=0; j<m; j++)
      {
         If(game[idx][1][i]==j)
            pay1[idx]= pay1[idx]*guess[i][j];
      {

   }

}
```

What I believe this function will do is set the vector pay 1 to be the building blocks needed for calculating each of the differential equations. So for our example game, if each player has an initial guess of playing each strategy 50% of the time then we will have:

initial pay1= {2,1,0.5,1}

After the function ExpectedPay

pay1 = {1, 0.5, 0.25, .5}

To determine the expected payoff for playing each pure strategy we need to determine which elements in the vector pay1 are associated with the same strategy and then take a parallel sum. I believe that Nvidia has ported the BLAS library over to CUDA so I won't try and determine the best way to do a parallel sum here.