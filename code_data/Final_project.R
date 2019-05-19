# NB: "CBM"--"can be modified"

rm(list = ls())

# set path where our data is stored
setwd("/Users/sophieyi/Academic/((Spring_2019))/MaL/news_project")

set.seed(1234567890)

######################################################################
##################### Package Setups #####################
######################################################################

#install.packages("tidytext")
#install.packages("topicmodels")
#install.packages("ldatuning")
#install.packages("stringi")
#install.packages("rjson")
#install.packages("doParallel")
#install.packages("stm")
#install.packages("geometry")
#install.packages("Rtsne")
#install.packages("rsvd")
#install.packages("bursts")
#install.packages("factoextra")
#install.packages("spacyr")
#install.packages("jiebaR")
#install.packages("rJST")

libraries <- c("ldatuning", "topicmodels", "ggplot2", "dplyr", "rjson", "caret",
               "quanteda", "lubridate", "parallel", "doParallel", "text2vec",
               "tidytext", "stringi", "tidyr", "stm", "geometry", "generics",
               "Rtsne", "rsvd", "quanteda.corpora", "readtext", "factoextra",
               "spacyr", "jiebaR", "tm", "stringr", "rJST", "randomForest")
lapply(libraries, require, character.only = TRUE)


######################################################################
##################### Preprocessing Texts #####################
######################################################################
mydata <- read.csv("df_airquality.csv", encoding = "UTF-8", colClasses = "character")

# tokenize Chinese texts
toks_mydata <- mydata
seg <- worker(symbol = F)

new_user_word(seg, c("pm"))
f<-readLines('stopwords_zh.txt') # read in the stopwords
stopwords<-c(NULL)
for(i in 1:length(f)){
  stopwords[i]<-f[i]
}

for (j in c("title", "content")){
  for (i in 1:2679){
  toks_mydata[i, j] <- seg[toks_mydata[i, j]] %>% filter_segment(stopwords) %>% str_c(collapse = " ")
  }
}

# make the key words consistent
toks_mydata$content <- gsub("PM2", "pm", toks_mydata$content)
toks_mydata$content <- gsub("PM1", "pm", toks_mydata$content)
toks_mydata$content <- gsub("PM", "pm", toks_mydata$content)
toks_mydata$content <- gsub("pm2", "pm", toks_mydata$content)
toks_mydata$content <- gsub("pm1", "pm", toks_mydata$content)

# get the vocabulary of the tokens
tokens <- toks_mydata$content %>% removeNumbers() %>% removePunctuation() # use the tokenized content from above
set.seed(1234567890)
tokens <- sample(tokens) # shuffle text

# create vocabulary
it <- itoken(tokens, progressbar = FALSE)
vocab <- create_vocabulary(it)


# create a dictionary of words
words <- as.list(vocab$term)
names(words) <- vocab$term
system.time(
  dict_zh <- dictionary(words)
)

# create corpus
corpus_mydata <- corpus(toks_mydata, 
                        docid_field = "title", 
                        text_field = "content") 

# words that should be removed
# stopw_zh <- stopwords("zh", source = "misc") # get Chinese stop words
# remove.words <- c(stopw_zh, "日", "月", "中", "新华社")
remove.words <- read.table("stopwords_zh.txt")

# create dfm
# remove customized stopwords, numbers, punctuations
system.time(
  dfm_mydata <- dfm(corpus_mydata, remove = remove.words$V1, remove_numbers = T, 
                  remove_punct = F, tolower = T, dictionary = dict_zh)
)
topfeatures(dfm_mydata)

# remove words that occur fewer than 30 times OR in fewer than 20 documents
dfm_mydata_trim <- dfm_mydata %>% 
  dfm_trim(min_termfreq = 30, min_docfreq = 20,     # NB: min_termfreq must <= MIN_COUNT above
           termfreq_type = "count", docfreq_type = "count")
topfeatures(dfm_mydata_trim)


######################################################################
##################### Estimate Topic Model #####################
######################################################################
set.seed(1234567890)
######### LDA model                 # CBM: use STM instead??
k <- 30 #set number of topics       # CBM: try different # of topics??
system.time(lda_tm <- LDA(dfm_mydata_trim, k = k, method = "Gibbs",  
                          control = list(iter = 2000, seed = 10012)))

# the most likely topic for each document
# lda_top_topic <- as.data.frame(topics(lda_tm))

######### Top 10 words for each topic
# Per topic per word proabilities matrix (beta)
lda_topics <- tidy(lda_tm, matrix = "beta") 
#lda_top_topic_frq <- table(lda_top_topic$`topics(lda_tm)`) %>%
#  sort(decreasing = T) %>% 
#  as.data.frame() %>%
#  print()
# head(lda_topics)

# Top 20 terms for each topic
lda_top_terms <- lda_topics %>%
  group_by(topic) %>%
  top_n(20, beta) %>%
  ungroup() %>%
  arrange(topic, -beta)
View(lda_top_terms)



######################################################################
######### record result for comparison ############
######################################################################
lda_tm_30 <- lda_tm # of topics = 30
lda_top_terms_30 <- lda_top_terms
lda_top_terms_30 <- lda_top_terms_30[with(lda_top_terms_30, order(topic, -beta)), , drop = T]
View(lda_top_terms_30)



######################################################################
######### Sentiment of the Skip-Gram of Topic Keywords ############
######################################################################
# choice parameters
WINDOW_SIZE <- 20
DIM <- 200
ITERS <- 10
MIN_COUNT <- 5
vocab <- prune_vocabulary(vocab, term_count_min = MIN_COUNT)  


######### Skip-Gram of each key word under each topic
# create term co-occurrence matrix
vectorizer <- vocab_vectorizer(vocab) 
tcm <- create_tcm(it, vectorizer, skip_grams_window = WINDOW_SIZE, 
                  skip_grams_window_context = "symmetric")

# estimate word embedding model
glove <- GlobalVectors$new(word_vectors_size = DIM,  #set model parameters
                           vocabulary = vocab, 
                           x_max = 100,
                           lambda = 1e-5)

word_vectors_main <- glove$fit_transform(tcm, 
                                         n_iter = ITERS,
                                         convergence_tol = 1e-3, 
                                         n_check_convergence = 1L,
                                         n_threads = RcppParallel::defaultNumThreads())

# get model output
word_vectors_context <- glove$components
word_vectors <- word_vectors_main + t(word_vectors_context) # word vectors

# get the word embeddings given by the model
nearest_neighbors <- function(cue, embeds, N, norm = "l2"){
  cos_sim <- sim2(x = embeds, y = embeds[cue, , drop = FALSE], method = "cosine", norm = norm)
  nn <- cos_sim <- cos_sim[order(-cos_sim),]
  return(names(nn)[2:(N + 1)])  # cue is always the nearest neighbor hence dropped
} # function to compute nearest neighbors

# get the word embeddings for each of the top key words # lda_top_terms_30[i, "term"]
lda_top_terms_30$term[grepl("奥会", lda_top_terms_30$term)==T] <- "冬奥会"
lda_top_terms_30$term[grepl("三角", lda_top_terms_30$term)==T] <- "珠三角"
lda_top_terms_30$term[grepl("标题", lda_top_terms_30$term)==T] <- "小标题"
embeddings <- matrix(NA, nrow = 600, ncol = 20)
for (i in 1:600){
  embeddings[i, ] <- nearest_neighbors(as.character(lda_top_terms_30[i, "term"]), word_vectors, N = 10, norm = "l2") %>% t()
}
embeddings <- data.frame(lda_top_terms_30, embeddings) # df of topic-specific terms and their word embeddings
View(embeddings)

######### For each word, the overall sentiment according to its word embedding(s) 
# embeddings are picked using skip-grams

# First, identify the key words that have clearcut sentiment labels and use them as the training set.
train_words_pos <- c("新", "改善", "推进", "保护", "技术", "科技", "利用", "应用", "科学", 
                     "提供", "措施", "发展", "改革", "民生", "保障", "服务", "更", "加强", 
                     "交流", "支持", "推动", "党", "人民", "创新", "坚持", "开放", "实现", "转型", 
                     "清洁", "天然", "生态", "建设", "绿色", "文明", "绿", "自然", "美丽")
train_words_neg <- c("污染", "重", "重度", "严重", "雾", "霾", "沙尘", "影响", "造成", "烟", "应", 
                     "发生", "尘", "天气", "问题", "健康", "导致", "口罩", "疾病", "污染物", "事故", 
                     "污", "情况", "违法", "落实")
train_set_pos <- NULL
train_set_neg <- NULL
test_set <- NULL
for (i in 1:600){
  if (is.element(embeddings$term[i], train_words_pos)==T){
    train_set_pos <- merge(train_set_pos, embeddings[i, ], all=T)
  }
  else if (is.element(embeddings$term[i], train_words_neg)==T){
    train_set_neg <- merge(train_set_neg, embeddings[i, ], all=T)
  }
  else {
    test_set <- merge(test_set, embeddings[i, ], all=T)
  }
}
train_set_pos$score <- 1
train_set_neg$score <- -1
train_set <- merge(train_set_pos, train_set_neg, all=T)
train_set <- train_set[order(-train_set$score), , drop = F]
View(train_set)
View(test_set)
#write.csv(train_set, file = "train_set.csv")

# Next, use the training set to identify the sentiment of the key words in the test set.
# Preprocessing: first create document feature matrix
train_set$words <- do.call(paste, c(train_set[, 4:23], sep = " ")) 
test_set$words <- do.call(paste, c(test_set[, 4:23], sep = " ")) 
dfm.train <- dfm(as.character(train_set$words), remove = remove.words$V1, remove_numbers = T, 
                 remove_punct = F, tolower = T, dictionary = dict_zh)
dfm.test <- dfm(as.character(test_set$words), remove = remove.words$V1, remove_numbers = T, 
                remove_punct = F, tolower = T, dictionary = dict_zh)
# match test set with training set
dfm.train <- dfm_match(dfm.train, featnames(dfm.test))
dfm.test <- dfm_match(dfm.test, featnames(dfm.train))

dfm.test <- convert(dfm.test, to = "matrix")
dfm.train <- convert(dfm.train, to = "matrix")

score.train <- train_set$score
score.test <- test_set$score

# prepare test and training sets
train.x <- dfm.train %>% as.data.frame() # train set data
train.y <- score.train %>% as.factor()  # train set labels
test.x <- dfm.test %>% as.data.frame() # test set data
test.y <- score.test %>% as.factor() # test set labels

# svm - linear
library(e1071)
set.seed(1234567890)
trctrl <- trainControl(method = "cv", number = 5)
svm.linear <- caret::train(x = train.x,
                    y = train.y,
                    method = "svmLinear",
                    trControl = trctrl)

svm.linear.pred <- predict(svm.linear, newdata = test.x)

# combine the scores into the results
test_set$score <- ifelse(svm.linear.pred=="-1", -1, 1)
embeddings_new <- merge(train_set, test_set, all=T)
embeddings_new <- embeddings_new[order(-embeddings_new$score), , drop = F]
View(embeddings_new)

######### Overall sentiment associated with each topic
# measured by the weighted average of the sentiments of all its key words
for (i in 1:30){
  weights <- embeddings_new[embeddings_new$topic == i, "beta"]/sum(embeddings_new[embeddings_new$topic == i, "beta"])
  sentiments <- embeddings_new[embeddings_new$topic == i, "score"]
  topic_sentiment <- t(sentiments) %*% weights
  embeddings_new$topic.score[embeddings_new$topic == i] <- topic_sentiment
}

embeddings_new <- embeddings_new[with(embeddings_new, order(topic, -beta)), , drop = T]
row.names(embeddings_new) <- NULL
View(embeddings_new)

######### Compare with human-coded results, JST
# results from above (reorganize)
topic_sentiment <- NULL
for (i in 0:30){
  m <- embeddings_new[embeddings_new$topic == i, c("topic", "term", "beta", "topic.score")]
  topic_sentiment <- merge(m, topic_sentiment, all=T)
}
topic_sentiment$sent <- ifelse(topic_sentiment$topic.score>-0.57735 & topic_sentiment$topic.score<0.57735, 0, 
                               ifelse(topic_sentiment$topic.score>0.57735, 1, -1))
View(topic_sentiment)

# human-coded
sent.hand <- c(1, 1, 0, 0, 1, 
               -1, -1, 1, 1, 0, 
               0, 1, 1, 1, 0, 
               0, -1, 1, 1, 0,
               0, 0, -1, 1, -1, 
               0, 0, 1, 1, 1)

# JST
set.seed(1234567890)
system.time(
  model.jst <- jst(dfm_mydata_trim, paradigm(), numSentiLabs = 3, numTopics = 30, numIters = 2000)
)

jst.topicsent <- model.jst@theta #topic-doc level sentiments
jst.topicsent.score <- colMeans(model.jst@theta[, -1])
jst.topicsent.score <- data.frame(jst.topicsent.score[1:30],
                                  jst.topicsent.score[31:60],
                                  jst.topicsent.score[61:90])
colnames(jst.topicsent.score) <- c("sentiment 1", "sentiment 2", "sentiment 3")
rownames(jst.topicsent.score) <- c(1:30)
View(jst.topicsent.score)
sent.jst <- colnames(jst.topicsent.score)[apply(jst.topicsent.score, 1, which.max)]
View(sent.jst)

for (i in 1:30){
  topic_sentiment$sent.hand[topic_sentiment$topic == i] <- sent.hand[i]
  topic_sentiment$sent.jst[topic_sentiment$topic == i] <- 
    ifelse(sent.jst[i]=="sentiment 1", 1, ifelse(sent.jst[i]=="sentiment 2", -1, 0)) %>%
    as.numeric()
}
topic_sentiment$sent.jst <- as.numeric(topic_sentiment$sent.jst)

# evaluate embedding-based and jst results, using hand-coded results as standard:
topic_sentiment$embed.vs.hand <- abs(topic_sentiment$sent-topic_sentiment$sent.hand)
topic_sentiment$jst.vs.hand <- abs(topic_sentiment$sent.jst-topic_sentiment$sent.hand)

# shorter version of the results
topic_sentiment.brief <- topic_sentiment[order(topic_sentiment[,'topic'],-topic_sentiment[,'beta']),]
topic_sentiment.brief <- topic_sentiment.brief[!duplicated(topic_sentiment.brief$topic),]
View(topic_sentiment.brief)
