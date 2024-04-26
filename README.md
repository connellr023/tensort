# $\text{tensort}\bullet\textit{(tensor-sort)}$
> A **CLI** tool that utilizes a **ResNet** convolutional neural network to recognize content in images and sort them into classes.

![Rust](https://img.shields.io/badge/rust-%23000000.svg?style=for-the-badge&logo=rust&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white)

<br />

### Usage
```
Usage: tensort <target_dir> <class_count> [-n | --no-names]

Arguments:s
<target_dir>    : Path to the target directory
<class_count>   : Number of classes
-n, --no-names  : Do not generate class names (optional)

Example:
tensort /path/to/images_dir 5 -n
```

The recognized image formats consist of:
`jpg`, `jpeg`, `png`

<br />

### How It Works
1. Read every image from `target_dir` and generate an embedding of each image. In the case of the pretrained model used in this application, the embedding is a `1000` dimensional vector representing a probabability distribution of likely classifications.

2. Compute pairwise cosine similarities for each embedding. This was done with the following formula. Let $t_0,  t_1$ be vectors in the same dimensional space, then, $$cs(t_0,  t_1) = \frac{t_0 \cdot t_1}{||t_0|| \times ||t_1||}$$ which produces a similarity value, $-1 \leq cs(t_0,  t_1) \leq 1$. Then, using this formula, pairwise cosine similarities are easily computed to produce the following cartesian relation, $$\langle  cs(t_0,  t_1),  cs(t_0,  t_2),  ...,  cs(t_0,  t_n),  ...,  cs(t_k,  t_0),  ...,  cs(t_k,  t_n)  \rangle$$ as a vector of similarity values. Now, extracting the cosine similarity between any two tensors in this vector, the following formula can be used, $$k = i + (j \times c)$$ where $k$ is the index of the target cosine similarity, $i$ is the index of the first tensor, $j$ is the index of the second tensor, and $c$ is the total number of tensors that embed images.

3. Generate similarity thresholds. I am using a heuristic algorithm which uses the calculated pairwise similarity vector and `class_count` to generate a similarity threshold which will be used to determine if an image belongs in one classification or should be in a new one. This similarity threshold is used to conduct initial class assignments in part 4.

4. Cluster image embeddings. Overall, this algorithm performs an initial assignment of embedding indices to clusters and then optimizes the assignment by finding the best fit for overflowed embedding indices based on cosine similarity.

5. Generate class names. This part can be opted out with the `-n | --no-names` flags mentioned above. In this part, a tensor averaged along each dimension is generated for each classification and then the classification with the highest probability is selected as the class name.

6. Finally, since everything has now been computed, moving the files into a directory tree that corresponds to the generated classifications is straightforward.

<br />

### Development Environment
The pretrained model used for this project can be found <a href="https://github.com/LaurentMazare/tch-rs/releases/download/mw/resnet18.ot">here</a>.

Setting up the development environment can be done by following the **README** from the **tch-rs** repository <a href="https://github.com/LaurentMazare/tch-rs">here</a>

When installing *libtorch*, ensure that the version that supports **CUDA** is used.

On Linux, `.bashrc` should contain the following (at least for my setup)
```
export LIBTORCH_BYPASS_VERSION_CHECK=1
export LIBTORCH=/path/to/libtorch
export LD_LIBRARY_PATH=/path/to/libtorch:$LD_LIBRARY_PATH
```

<br />
<br />

<div align="center">
  Developed and Tested by <b>Connell Reffo</b> in <b>2024</b>.
</div>
