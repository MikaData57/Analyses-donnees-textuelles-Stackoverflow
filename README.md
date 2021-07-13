![intro](http://www.mf-data-science.fr/images/projects/intro.jpg)

# Machine Learning - Stackoverflow tags generator

## Table of contents
* [General information](#general-info)
* [Data](#data)
* [Technologies](#technologies)
* [Setup](#setup)
* [API](#API)

## <span id="general-info">General information</span>
Machine Learning algorithm designed to automatically assign several relevant tags to a question asked on the famous Stack overflow site.     
This program is mainly intended for new users, in order to suggest some tags relating to the question they wish to ask.

## <span id="data">Data</span>
The data was captured using the [stackexchange explorer](https://data.stackexchange.com/) data export tool, which collects a large amount of genuine data from the peer support platform.
They relate to the period 2009/2020 and only to "quality" posts with at least 1 response, 5 comments, 20 views and a score greater than 5.

```SQL
DECLARE @start_date DATE
DECLARE @end_date DATE
SET @start_date = '2011-01-01'
SET @end_date = DATEADD(m , 12 , @start_date)

SELECT p.Id, p.CreationDate, p.Title, p.Body, p.Tags,
p.ViewCount, p.CommentCount, p.AnswerCount, p.Score 
FROM Posts as p
LEFT JOIN PostTypes as t ON p.PostTypeId = t.id
WHERE p.CreationDate between @start_date and @end_date
AND t.Name = 'Question'
AND p.ViewCount > 20
AND p.CommentCount > 5
AND p.AnswerCount > 1
AND p.Score > 5
AND len(p.Tags) > 0
```
	
## <span id="technologies">Technologies</span>
Project is created with:
* [Kaggle](https://www.kaggle.com/michaelfumery) Notebook
* Python 3.8 *(Numpy, Pandas, Sklearn, NLTK ...)*

	
## <span id="setup">Setup</span>
Download the Notebook and import it preferably into Google Colaboratoty, Kaggle or Jupyter via Anaconda.      
Then just perform a *Run all* to run the project.

## <span id="API">API</span>
An API developed with Flask and Swagger based on the logistic regression model is available in a second Git repostory at: https://github.com/MikaData57/stackoverflow_swagger_api