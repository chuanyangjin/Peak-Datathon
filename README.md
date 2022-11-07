# Peak-Datathon
Team Members: Charlie Chen, Chuanyang Jin, [Gavin Yang](https://github.com/redagavin), Hongyi Zheng (alphabetical order)

This is our team project for NYU x Peak Datathon 2022. Check our slides for a non-technical illustration and our notebook for implementation details!


### Problem Statement
In this challenge, we are asked to build a recommendation system from the [dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) provided by Olist.

As we will show more details in the following sections, only 10k users out of 100k users in the dataset buy two or more items. What's more, the side information for users and items are limited. This makes the classical machine learning recommendation pipelines especially hard because they generally depends on sequencial purchase history and user profiles (which we only have geospacial information). So, it is important to formally define our own statement for the challenge and how we will resolve this problem.

For the sake of sanity, we will use $prod$ to denote products, $cust$ to denote customers, $purch$ to denote purchase action. The essense of recommendation system is to solve the formula $$\hat{prod} = argmax_{prod} P(purch \mid prod, cust)$$ So, **we target to solve this formula by exploiting the dataset with our best effort**. Since there are only ~10k users who have 2 or more purchase history, we will based our evaluation solely based on them.

#### Counter justification for our problem statement
**Why not state the problem as a cold-start problem?**

If we don't take into the purchase history into considerations, we only have geospacial information, which is not sufficient for recommendation.

### Framework to Solve the Statement
We target to solve the problem from **a probablistic point of view with the help of deep learning**. To be precise,

$$
P(purch \mid prod, cust) \propto P(purch \mid \text{cust location}, prod) P(purch \mid \text{cust past purchase prod}, prod) P(purch \mid prod)
$$

The first term is the **customer-product relation**. The second term is the **product-product relation**. The third term can be just interpreted as the **intrinsic product features**.


### Part 1: Bayesian Inference (Customer-Product Relation)
We hope to recommend products that are either:

- Popular in general, or

- Especially favored by customers from the same state.

Assuming whether the user like the product is positively correlated with whether the user will buy the product, we deduce that $P(location|purchase)$ is non-trivial, so we can construct the Bayesian classifier to recommend the users with the products they are most likely to purchase.

$$
\begin{aligned}
P(like|location) &= \frac{P(location | like) * P(like)} {\sum_{like} P(location | like) * P(like)}
\end{aligned}
$$

### Part 2: Category Similarity (Product-Product Relation)

We only have very limited customers who purchase more than one items, but we have plenty of vendors selling multiple products.
This motivates us to measure category similarities based on the assumptions that if two categories of products are sold by the same seller, they tends to be more similar.

We construct an embedding for each category via Item2Vec, and train the embeddings so that the cosine similarity of two embeddings represents similarity between two categories.


### Part 3: Incorporating Other Features (Intrinsic Product Features)
Apart from the two metrics above, we argue the following regarding the recommendation score:

1. A customer is more willing to buy a product with a higher average rating.

2. As we have shown before, the more distance between the customer and the product's seller, the less likely the customer would buy the product. Therefore, the score should be negatively correlated with the distance. We analyze the correlation and choose to use a log scale.

3. A customer is more willing to buy a product of similar price with his previous purchase. We represent the price difference by comparing the ratio between $\text{price}$ and $\text{previous price}$ in a log scale. Under this assumption, a customer will be equally likely to be recommended a product with the price of $3 \times \text{previous price}$ and another product with the price of $\frac{1}{3} \times \text{previous price}$.


### Final Part: Ensemble Model
To make use of all the above metrics, we propose a score formula as follows:
$$\text{final score} = \alpha_1 \cdot \text{bayesian inference score} + \alpha_2 \cdot \text{category similarity score} + \alpha_3 \cdot \text{product average rating} - \alpha_4 \cdot \text{distance metrics} - \alpha_5 \cdot \text{price difference metrics}$$
where

$\text{bayesian inference score}$ is calculated from Part 1,

$\text{category similarity score}$ is calculated from Part 2,

$\text{product average rating}=\frac{1}{\text{num of product ratings}}\sum_{\text{rating} \in \text{product ratings}} \text{ratings}$,

$\text{distance metrics} = \log(\text{distance})$,

$\text{price difference metrics} = \left|\log\left(\frac{\text{price}}{\text{previous price}}\right)\right|$,

$\alpha_i, 1 \leq i \leq 5$ are hyperparamters.

We perform the grid search for the optimal hyperparameters.
