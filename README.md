# Contextualised-Image-Classifiers

This repository contains the majority of the code that I produced for my undergraduate dissertation. The rest of the code is in the [OpenImagesV5Tools](https://github.com/KieranLitschel/OpenImagesV5Tools) library I created. 

I was awarded a grade of First Class for the dissertation, which you can read [here](https://github.com/KieranLitschel/Contextualised-Image-Classifiers/blob/master/Honors_Project_Report.pdf). 

## Abstract

Mahajan et al. and Yalniz et al. demonstrate the beneﬁt of pre-training on large unlabeled image datasets using user tags as labels for weak-supervision. We take this approach further, considering whether the use of user tags in large labeled image datasets is beneﬁcial.

We propose modifying the teacher in the semi-weakly supervised approach proposed by Yalniz et al. to include user tags as features. To enable this we constructed a new dataset with user tags and human-veriﬁed labels, by combining the Open Images dataset and Yahoo Flickr Creative Commons 100 Million dataset.

We were unable to combine user tags with image classiﬁers due to engineering challenges that we were unable to solve due to the time constraints. Instead we experimented with classifying images using only their user tags. Despite this, our model outperforms our naive baseline, and achieves a 0.1 higher average AP than our image classiﬁer baseline for some individual classes, most notably ”animal”. This suggests that the use of user tags as features for image classiﬁers may help to improve performance for at least some individual classes, warranting further research.

## Reflections on the project a few months on

This project turned out much better than my [first machine learning project](https://github.com/KieranLitschel/PredictingClosingPriceTomorrow), but I still could have approached the project better. I recently completed Andrew Ng’s Deep Learning Specialization, and the [Structuring Machine Learning Projects](https://www.coursera.org/learn/machine-learning-projects/) course has helped me identify some of the mistakes I made.

The biggest mistake I made in this project was not approaching it iteratively. After deciding on the topic of the dissertation and my approach to solving it, I should have trained the most basic model I proposed (the image classifier with the user tag feature encoder). I then should have randomly sampled some images misclassified by the model and performed error analysis on them to look for patterns in the mistakes. This would have helped me identify the edge cases the model struggled to classify and prioritise which were most important to solve.

What I actually did after deciding on the topic and approach, was spend a lot of time thinking about the potential edge cases of the model (for example, those arising from the multilingual corpus), and researching how I might solve them. This meant that I did most of my thinking before implementing anything. When I did start the implementation, I implemented in parallel the basic model and the solutions to the edge cases. The problem with this approach was I did not have time to write about a lot of the solutions I came up with in the dissertation, let alone run experiments for them. I ended up writing a lot of code that I never used, only running experiments for the basic model and pre-training. The implementation took longer than expected, and I encountered big engineering challenges, which resulted in a several month code crunch. This ultimately made the project much more stressful than it needed to be.

If I had taken an iterative approach, the project would have turned out much better, and would have been much less stressful. I would have encountered the big engineering challenges early on and would have had time to solve them. Fully implementing the basic model would have allowed me to test my initial hypothesis. From there, I could have worked on improving the model. Prioritising the edge cases of the initial model in order of how common they were, and how easy they were to solve, and then solving them one at a time. This would have avoided the code crunch, as I could have just stopped iterating if I ran out of time.

Another big mistake I made was not performing bias and variance analysis. Solving problems with bias and variance individually in an iterative fashion would have reduced the number of hyperparameters I needed to tune at one time, making the search much faster.
