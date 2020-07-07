# Contextualised-Image-Classifiers

This repository contains the majority of the code that I produced for my undergraduate dissertation. The rest of the code is in the [OpenImagesV5Tools](https://github.com/KieranLitschel/OpenImagesV5Tools) library I created. 

I was awarded a grade of First Class for the dissertation, which you can read [here](https://github.com/KieranLitschel/Contextualised-Image-Classifiers/blob/master/Honors_Project_Report.pdf). 

## Abstract

Mahajan et al. and Yalniz et al. demonstrate the beneﬁt of pre-training on large unsupervised image datasets using user tags as labels for weak-supervision. We take this approach further, considering whether the use of user tags in large supervised image datasets is beneﬁcial.

We propose modifying the teacher in the semi-weakly supervised approach proposed by Yalniz et al. to include user tags as features. To enable this we constructed a new dataset with user tags and human-veriﬁed labels, by combining the Open Images dataset and Yahoo Flickr Creative Commons 100 Million dataset.

We were unable to combine user tags with image classiﬁers due to engineering challenges that we were unable to solve due to the time constraints. Instead we experimented with classifying images using only their user tags. Despite this, our model outperforms our naive baseline, and achieves a 0.1 higher average AP than our image classiﬁer baseline for some individual classes, most notably ”animal”. This suggests that the use of user tags as features for image classiﬁers may help to improve performance for at least some individual classes, warranting further research.
