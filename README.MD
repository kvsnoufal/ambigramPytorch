# GitHub Repository Readme.md


## Generating Ambigrams using Deep Learning: A Typography Approach
Developing a typography technique that creates path-vectors given a text, and then runs an optimization routine using pytorch on the vectors to generate an ambigram.

Full article : https://medium.com/mlearning-ai/generating-ambigrams-using-deep-learning-a-typography-approach-c829d0ee4d51

In this article, I explore an approach to generating ambigrams using deep learning techniques. We use pytorch for implementing our optimizer and nueral network models.

## Approach
This approach involves generating vector images by using a chosen font and text, preserving gradients through differential rasterization. The rasterized image is then used to extract embeddings with a pretrained OCR model. Through defining a loss function and optimization, the vector image's points are adjusted to achieve rotational symmetry and preserve text content, resulting in refined ambigram structures.

![Image 1](https://github.com/kvsnoufal/ambigramPytorch/blob/main/docs/img1.png)
1)Given a font and an input text, create a vector image of the text using the font. The output is a svg file. 2) Rasterize the vector image using diffsvg - diffsvg is a differential rasterizer and hence preserves the gradients during rasterization process. 3) Use a pretrained ocr model to get the embeddings of the rasterized image. 4) Calculate loss function that achieves 2 things.i) The image and its 180 deg rotation should be identical.ii) The text content in the image should not change. This is achieved by comparing ocr embeddings of the input image, with those of the candidate embeddings.5) Optimize the vector image (some points) to reduce the loss function


## Results
You can also embed YouTube videos directly in your readme file. Here's an example of embedding a video:

![Image 1](https://github.com/kvsnoufal/ambigramPytorch/blob/main/docs/gif1.gif)

![Image 1](https://github.com/kvsnoufal/ambigramPytorch/blob/main/docs/gif2.gif)


## Shoulders of Giants
1. DiffSVG: https://github.com/BachiLi/diffvg
2. Word-As-Image: https://github.com/Shiriluz/Word-As-Image
3. EazyOCR: https://github.com/JaidedAI/EasyOCR
