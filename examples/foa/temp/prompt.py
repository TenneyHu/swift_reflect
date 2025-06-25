# 5 shot
standard_prompt = '''Solve mathematical reasoning questions.

Question:
How many vertical asymptotes does the graph of $y=\\frac{2}{x^2+x-6}$ have?

Answer:
2

Question:
What is the positive difference between $120\\%$ of 30 and $130\\%$ of 20?

Answer:
10

Question: 
Find $x$ such that $\\lceil x \\rceil + x = \\dfrac{23}{7}$. Express $x$ as a common fraction.

Answer:
\\frac{9}{7}

Question:
Evaluate $i^5+i^{-25}+i^{45}$.

Answer:
i

Question:
If $2^8=4^x$, what is the value of $x$?

Answer:
4

Question:
{question}

Answer:
'''

cot_prompt = '''Solve mathematical reasoning questions. Please reason step by step and explain how you got to the solution.

Question:
How many vertical asymptotes does the graph of $y=\\frac{2}{x^2+x-6}$ have? 

Thoughts:
The denominator of the rational function factors into $x^2+x-6=(x-2)(x+3)$. Since the numerator is always nonzero, there is a vertical asymptote whenever the denominator is $0$, which occurs for $x = 2$ and $x = -3$. Therefore, the graph has 2 vertical asymptotes.

Answer:
2

Question:
What is the positive difference between $120\\%$ of 30 and $130\\%$ of 20?

Thoughts:
One hundred twenty percent of 30 is $120\\cdot30\\cdot\\frac{1}{100}=36$, and $130\\%$ of 20 is $ 130\\cdot 20\\cdot\\frac{1}{100}=26$. The difference between 36 and 26 is 10.

Answer:
10

Question: 
Find $x$ such that $\\lceil x \\rceil + x = \\dfrac{23}{7}$. Express $x$ as a common fraction.

Thoughts:
First, we note that $x$ must be positive, since otherwise $\\lceil x \\rceil + x$ is nonpositive. Next, we know that the decimal part of $x$ must be $\\dfrac{2}{7}$. We write $x$ as $n+\\dfrac{2}{7}$, where $n$ is the greatest integer less than $x$. Then, $\\lceil x \\rceil = n + 1$. Therefore, we can write $\\lceil x \\rceil + x$ as $n+1+n+\\dfrac{2}{7}=\\dfrac{23}{7}$. Solving, we get $n=1$. Therefore, the only value $x$ that satisfies the equation is $1+\\dfrac{2}{7}=\\frac{9}{7}$.

Answer:
\\frac{9}{7}

Question:
Evaluate $i^5+i^{-25}+i^{45}$.

Thoughts:
We have $i^5 = i^4\\cdot i = 1\\cdot (i) = i$.  We also have $i^{-25} = 1/i^{25} = 1/(i^{24}\\cdot i) = 1/[1\\cdot (i)] = 1/i = \\frac1{i}\\cdot\\frac{i}{i} = i/(-1) = -i$ and $i^{45} = (i^{44})\\cdot i= 1\\cdot i =i$. So, adding these three results gives $i^5 + i^{-25} + i^{45} = i+-i+i = i$.

Answer:
i

Question:
If $2^8=4^x$, what is the value of $x$?

Thoughts:
Rewrite $4$ as $2^2$ to find $4^x=2^{2x}$. Since $2^8=2^{2x}$, we have $2x=8$ which implies $x=4$.

Answer:
4

Question:
{question}
'''

propose_prompt = '''Let's solve a mathematical reasoning question. Please provide a step-by-step solution.

Question:
{question}
Current status:
{reasoning_chain}

Given the current status, list all possible next reasoning steps, and your confidence levels (certain/high/medium/low), using the format "reasoning step (medium)". Use "certain" cautiously and only when you are 100% sure this is the correct word.
'''

value_prompt = '''Evaluate how good the reasoning chain is for solving the mathematical reasoning question. To indicate the likelihood of reaching the correct final answer from the reasoning chain choose one word from (sure/maybe/impossible).

Question:
How many vertical asymptotes does the graph of $y=\\frac{2}{x^2+x-6}$ have?
Reasoning chain:
The denominator of the rational function factors into $x^2+x-6=(x-2)(x+3)$. Since the numerator is always nonzero, there is a vertical asymptote whenever the denominator is $0$, which occurs for $x = 2$ and $x = -3$. Therefore, the graph has 2 vertical asymptotes.
Judge:
sure

Question:
What is the positive difference between $120\\%$ of 30 and $130\\%$ of 20?
Reasoning chain:
One hundred twenty percent of 30 is $120\\cdot30\\cdot\\frac{1}{100}=360$, and $130\\%$ of 20 is $ 130\\cdot 20\\cdot\\frac{1}{100}=260$. 
Judge:
impossible 

Question: 
Find $x$ such that $\\lceil x \\rceil + x = \\dfrac{23}{7}$. Express $x$ as a common fraction.
Reasoning chain:
First, we note that $x$ must be positive, since otherwise $\\lceil x \\rceil + x$ is nonpositive. Next, we know that the decimal part of $x$ must be $\\dfrac{2}{7}$. We write $x$ as $n+\\dfrac{2}{7}$, where $n$ is the greatest integer less than $x$. Then, $\\lceil x \\rceil = n + 1$. Therefore, we can write $\\lceil x \\rceil + x$ as $n+1+n+\\dfrac{2}{7}=\\dfrac{23}{7}$. Solving, we get $n=1$ Therefore, the only value $x$ that satisfies the equation is $1+\\dfrac{2}{7}=\\frac{9}{7}.
Judge:
sure

Question:
Evaluate $i^5+i^{-25}+i^{45}$.
Reasoning chain:
We have $i^5 = i^4\\cdot i = 1\\cdot (i) = i$.
Judge:
maybe

Question:
If $2^8=4^x$, what is the value of $x$?
Reasoning chain:
Rewrite $4$ as $2^2$ to find $4^x=2^{2x}$. Since $2^8=2^{2x}$, we have $2x=8$ which implies $x=4$.
We have that $2^8=4^x$, so $(2^2)^4=2^{2x}$.  Therefore, $2^4=2^{2x}$, so $4=2x$.
Judge:
impossible

Question:
{question}
Reasoning chain:
{reasoning_chain}
Judge:
'''

foa_step_prompt = '''Obtain the next step in the reasoning chain to solve the mathematical reasoning question. Perform only one step.

Question:
Evaluate $i^5+i^{-25}+i^{45}$.
Reasoning chain:
We have $i^5 = i^4\\cdot i = 1\\cdot (i) = i$.
Possible next step:
We also have $i^{-25} = 1/i^{25} = 1/(i^{24}\\cdot i) = 1/[1\\cdot (i)] = 1/i = \\frac1{i}\\cdot\\frac{i}{i} = i/(-1) = -i$.

Question:
What is the positive difference between $120\\%$ of 30 and $130\\%$ of 20?
Reasoning chain:
One hundred twenty percent of 30 is $120\\cdot30\\cdot\\frac{1}{100}=36$, and $130\\%$ of 20 is $ 130\\cdot 20\\cdot\\frac{1}{100}=26$.
Possible next step:
The difference between 36 and 26 is 10.

Question:
{question}
Reasoning chain:
{reasoning_chain}
Possible next step:
'''
