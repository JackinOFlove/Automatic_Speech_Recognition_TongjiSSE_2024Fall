# Assignment2: Viterbi Algorithm

[TOC]

## 1. Problem Description 

+ **Teacher-mood-model**

 One week, your teacher gave the following homework assignments:

| **Monday** | **Tuesday** | **Wednesday** | **Thursday** | **Friday** |
| ---------- | ----------- | ------------- | ------------ | ---------- |
| A          | C           | B             | A            | C          |

+ **Questions**

What did his mood curve look like most likely that week?

Give the full process of computation in your report.

You can refer to the Result Table in P. 60 in ch3 HMM.pptx. p.s. Some numbers in the table are incorrect.

## 2. Computing Process 

Following the prompts on the PPT, we can set the following parameters and draw the following table:

<img src=".\assets\ModelParameters.png" alt="ModelParameters" style="zoom:30%;" />

<img src=".\assets\MoodSwitch.png" alt="MoodSwitch" style="zoom:50%;" />

According to the above chart of PPT, the state transfer probability distribution matrix is:	$A= \begin{bmatrix}  0.2 & 0.3 & 0.5 \\  0.2 & 0.2 & 0.6 \\  0.0 & 0.2 & 0.8 \\ \end{bmatrix} $

and the observed state probability matrix is:	$B=\begin{bmatrix}  0.7 & 0.2 & 0.1\\  0.3 & 0.4 & 0.3 \\  0.0 & 0.1 & 0.9 \\ \end{bmatrix}$

I define $\Phi_{t}(j)=arg\ \underset {i \in S}{max}(V^{i}_{t-1}$\*$a_{ij})$ ,$(i,j, \Phi_t(j) \in S=\{good, neutral, bad\})$, is the node passing at the previous moment of the probability maximization path from time $t$ to state $j$, which retains the node passing through the shortest path.I set it here for the mood of the one day before.

+ **Empty table**

|             | A    | C    | B    | A    | C    |
| ----------- | ---- | ---- | ---- | ---- | ---- |
| **good**    |      |      |      |      |      |
| **neutral** |      |      |      |      |      |
| **bad**     |      |      |      |      |      |



I assume here that the three states are equally likely to appear.

So the initial state is set to:   $\Pi=(\frac{1}{3},\frac{1}{3},\frac{1}{3})$.

+ **Initialization: $V^{j}_{1}=b_{j}(x_{1})p(q_{1}=j)=\frac{b_{j}(x_{1})} {\# states}$**(The formula in PPT in class)

so, we can get that:

$b_{good}(A)=0.7,b_{neutral}(A)=0.3,b_{bad}(A)=0.0$ 

$V^{good}_{1}=b_{good}(A)p(q_{1}=good)=\frac{b_{good}(A)} {\# states}=\frac{0.7}{3}=0.2\dot3\approx0.23$     (After it, I calculated it at 0.23)

$V^{neutral}_{1}=b_{neutral}(A)p(q_{1}=neutral)=\frac{b_{neutral}(A)} {\# states}=\frac{0.3}{3}=0.1$

$V^{bad}_{1}=b_{bad}(A)p(q_{1}=bad)=\frac{b_{bad}(A)} {\# states}=\frac{0.0}{3}=0.0$

Here we might as well make $\Phi_{1}(good)=\Phi_{1}(neutral)=\Phi_{1}(bad)=0$. There is no meaning here.

Fill in the form:

|             | A    | C    | B    | A    | C    |
| ----------- | ---- | ---- | ---- | ---- | ---- |
| **good**    | 0.23 |      |      |      |      |
| **neutral** | 0.1  |      |      |      |      |
| **bad**     | 0.0  |      |      |      |      |



+ **Iteration: $V^{j}_{t}=b_{j}(x_{t})max(V^{i}_{t-1}$\*$a_{ij})$ for all states $i,j \in S,t\geq 2$.** (The formula in PPT in class)

+ **When t = 2: $V^{j}_{2}=b_{j}(C)max(V^{i}_{1}$\*$a_{ij})$ **

$b_{good}(C)=0.1,b_{neutral}(C)=0.3,b_{bad}(C)=0.9$ 

$V^{good}_{2}=b_{good}(C)max(0.23*0.2,0.1*0.2,0.0*0.0)=0.1* 0.046=0.0046$

 $i=1,$ so $ \Phi_{2}(good)=good$.

$V^{neutral}_{2}=b_{neutral}(C)max(0.23*0.3,0.1*0.2,0.0*0.2)=0.3* 0.069=0.0207$

 $i=1,$ so $\Phi_{2}(neutral)=good$.

$V^{bad}_{2}=b_{bad}(C)max(0.23*0.5,0.1*0.6,0.0*0.8)=0.9* 0.115=0.1035$

 $i=1,$ so $\Phi_{2}(bad)=good$.

Fill in the form:

|             | A    | C      | B    | A    | C    |
| ----------- | ---- | ------ | ---- | ---- | ---- |
| **good**    | 0.23 | 0.0046 |      |      |      |
| **neutral** | 0.1  | 0.0207 |      |      |      |
| **bad**     | 0.0  | 0.1035 |      |      |      |



+ **When t = 3: $V^{j}_{3}=b_{j}(B)max(V^{i}_{2}$\*$a_{ij})$ **

$b_{good}(B)=0.2,b_{neutral}(B)=0.4,b_{bad}(B)=0.1$ 

$V^{good}_{3}=b_{good}(B)max(0.0046*0.2,0.0207*0.2,0.105*0.0)=0.2*0.0207*0.2=0.000828$

$i=2,$ so $\Phi_{3}(good)=neutral$.

$V^{neutral}_{3}=b_{neutral}(B)max(0.0046*0.3,0.0207*0.2,0.1035*0.2)=0.4*0.1035*0.2=0.00828$

$i=3,$ so $\Phi_{3}(neutral)=bad$.

$V^{bad}_{3}=b_{bad}(B)max(0.0046*0.5,0.0207*0.6,0.1035*0.8)=0.1*0.1035*0.8=0.00828$

$i=3,$ so $\Phi_{3}(bad)=bad$.

Fill in the form:

|             | A    | C      | B        | A    | C    |
| ----------- | ---- | ------ | -------- | ---- | ---- |
| **good**    | 0.23 | 0.0046 | 0.000828 |      |      |
| **neutral** | 0.1  | 0.0207 | 0.00828  |      |      |
| **bad**     | 0.0  | 0.1035 | 0.00828  |      |      |



+ **When t = 4: $V^{j}_{4}=b_{j}(A)max(V^{i}_{3}$\*$a_{ij})$ **

$b_{good}(A)=0.7,b_{neutral}(A)=0.3,b_{bad}(A)=0.0$ 

$V^{good}_{4}=b_{good}(A)max(0.000828*0.2,0.00828*0.2,0.00828*0.0)=0.7*0.00828*0.2=0.0011592$

 $i=2,$ so $\Phi_{4}(good)=neutral$.

$V^{neutral}_{4}=b_{neutral}(A)max(0.000828*0.3,0.00828*0.2,0.00828*0.2)=0.3*0.00828*0.2=0.0004968$

 $i=2,$ so $\Phi_{4}(neutral)=neutral$

$V^{bad}_{4}=b_{bad}(A)max(0.000828 *0.5,0.00828*0.6,0.00828*0.8)=0.0$

 $i=3,$ so $\Phi_{4}(bad)=bad$.

Fill in the form:

|             | A    | C      | B        | A         | C    |
| ----------- | ---- | ------ | -------- | --------- | ---- |
| **good**    | 0.23 | 0.0046 | 0.000828 | 0.0011592 |      |
| **neutral** | 0.1  | 0.0207 | 0.00828  | 0.0004968 |      |
| **bad**     | 0.0  | 0.1035 | 0.00828  | 0.0       |      |

<div style="page-break-after: always;"></div>

+ **When t = 5: $V^{j}_{5}=b_{j}(A)max(V^{i}_{4}$\*$a_{ij})$ **

$b_{good}(C)=0.1,b_{neutral}(C)=0.3,b_{bad}(C)=0.9$ 

$V^{good}_{5}=b_{good}(C)max(0.0011592*0.2,0.0004968*0.2,0.0*0.0)=0.1*0.0011592*0.2=0.000023184$

 $i=1,$ so $\Phi_{5}(good)=good$.

$V^{neutral}_{5}=b_{neutral}(C)max(0.0011592*0.3,0.0004968*0.2,0.0*0.2)=0.3*0.0011592*0.3=0.000104328$

 $i=1,$ so $\Phi_{5}(neutral)=good$.

$V^{bad}_{5}=b_{bad}(C)max(0.0011592 *0.5,0.0004968*0.6,0.0*0.8)=0.9*0.0011592*0.5=0.00052164$

$i=1,$ so $\Phi_{5}(bad)=good$.

Fill in the form:

|             | A    | C      | B        | A         | C           |
| ----------- | ---- | ------ | -------- | --------- | ----------- |
| **good**    | 0.23 | 0.0046 | 0.000828 | 0.0011592 | 0.000023184 |
| **neutral** | 0.1  | 0.0207 | 0.00828  | 0.0004968 | 0.000104328 |
| **bad**     | 0.0  | 0.1035 | 0.00828  | 0.0       | 0.00052164  |



The maximum probability at $t=5$ is at $V_5^{bad}$(**bad-C** is the last state), Let's look back:

+  $\Phi_5(bad)=good$, that is at $t=4$, the mood is most probably **good**;
+ $\Phi_4(good)=neutral$,  that is at $t=3$, the mood is most probably **neutral**;
+ $\Phi_3(neutral)=bad$,  that is at $t=2$, the mood is most probably **bad**;
+ $\Phi_2(bad)=good$,  that is at $t=1$, the mood is most probably **good**;

At last, Reconstruct path along pointers, we can get(The path is a bold font):

|             | A        | C          | B           | A             | C              |
| ----------- | -------- | ---------- | ----------- | ------------- | -------------- |
| **good**    | **0.23** | 0.0046     | 0.000828    | **0.0011592** | 0.000023184    |
| **neutral** | 0.1      | 0.0207     | **0.00828** | 0.0004968     | 0.000104328    |
| **bad**     | 0.0      | **0.1035** | 0.00828     | 0.0           | **0.00052164** |

His mood curve look like most likely that week: 

| Monday | **Tuesday** | **Wednesday** | **Thursday** | **Friday** |
| ------ | ----------- | ------------- | ------------ | ---------- |
| A      | C           | B             | A            | C          |
| good   | bad         | neutral       | good         | bad        |

