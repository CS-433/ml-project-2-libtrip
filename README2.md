We are working on a project which focusing analysing idea generation within a venture team in terms of team members experience.

The data set is initially composed of 117 videos of 1h and is a multimodel data as it combines audio, transcript and videos.

Our collaboration in this research focuses on determining either individuals within a venture team are more individualist or collectivist persons. It was proven by many authors (....) that this type of personnality can impact the behaviour of a person espacially within a group, this analysis could help us find or refute a correlation between indivualistic and collectivistic persons or a causality relation.

Our research question : How does the presence of individualistic or collectivistic behavior within startup teams correlate with the frequency and quality of generated innovative ideas during brainstorming sessions?

As we don't have a lot of time we decided to focus only on what aspect of our research question which is the classification of individuals within a team as individualistic or collectivistic.

After reading a bunch of paper and inferring (X) informations on the subject of individualist or collectivistic we faced many challenges. One of the biggest challenges was to find a method to perform a classification through machine learning. Fisrt we noticed that all studies which made classification consist in psychological studies and that a few papers focused on this subject in ML thus we decided to select on of the psychology scales and use it with data and computers! We also find a study on forum(stack overflow) which did a kind of similar analysis in another context regarding people who post on forum, they classified the persons into three types depending on their country and categorized them into ind. or coll. using Hofsted approach ou analysis . They find interesitng result that we will use to perform our predictions.

In this study we do not have a way to test our model as we will perform unsupervised learning on textual data. Thus we choose to work on two different approaches for classifing our team members :

1. Approach 1 :

We will create many features that we will use to do a classification :

- personnal pronoun use (using scipy) // calculer le nombre de pronom + verb / I ou We
- future/past verbs ( scipy) // + verb futur / present
- direct/indirect communication ( find a library that can do sentiment analysis in german)
- postive/negative emotions ??

et faire une classification en choisissant notre theshold, ou méthode ou paramètre pour determiner si UNE PERSONNE dans le groupe est indiv ou collectif —> MORPHOLOGICAL ANALYSIS

2. Approach 2:

We will use a large language model such as (LAMA , chatGPT) to compute INDCOL scale according to (... ) work .

Compute Vertical collectivism, Horizontal collectivism, Vertical Individualism, Horizontal individualism

Each value will be processed to measure the scale and then determining if a team member is individualis or not.
TO do that we will train the LLM with concrete examples of scales feature and thun plug in our data so that it can make the classification for us. This replaces the process of reading the transcripts and give a grade by hand.

-->for validation, PCA between models
