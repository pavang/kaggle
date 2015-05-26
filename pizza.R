setwd("C:/kaggle/pizza")
options(java.home="C:/Program Files/Java/jre1.8.0_45")
library("jsonlite")
library("openNLP")
library("NLP")
library("tm")
library("stringi")
library("slam")
library("topicmodels")
library("ggplot2")
library("caret")
library("randomForest")
library("e1071")
library("reshape2")
library("pROC")
library("gbm")

# library("infotheo")
# library("corrplot")

train.data <- fromJSON("train.json")
test.data<- fromJSON("test.json")

master.train.data<-train.data
master.test.data<-test.data

dict.data<-read.csv("feelings.csv",header=TRUE,stringsAsFactors=FALSE)

createDocTermMatrix <- function(text.data) {
  docs<-Corpus(VectorSource((text.data)))
  docs2 <- tm_map(docs, function(x) stri_replace_all_regex(as.character(x), "<.+?>", " "))
  docs3 <- tm_map(docs2, function(x) stri_replace_all_fixed(x, "\t", " "))
  docs4 <- tm_map(docs3, PlainTextDocument)
  docs5 <- tm_map(docs4, stripWhitespace)
  docs6 <- tm_map(docs5, removeWords, stopwords("english"))
  docs7 <- tm_map(docs6, removePunctuation)
  docs8 <- tm_map(docs7, tolower)
  docs9 <- tm_map(docs8, stemDocument)
  docs9 <- tm_map(docs9, PlainTextDocument)
  dict.data<-read.csv("feelings.csv",header=TRUE,stringsAsFactors=FALSE)
  docterm.matrix<-DocumentTermMatrix(docs9,control=list(dictionary=dict.data[,1]))
  term.matrix<-as.data.frame(as.matrix(docterm.matrix))
  term.matrix<-t(term.matrix)
  term.matrix<-data.frame(term.matrix,dict.data[,2])
  agg.matrix<-as.data.frame(aggregate(term.matrix[,1:ncol(term.matrix)-1],by=list(term.matrix[,ncol(term.matrix)]),sum),stringsAsFactors=FALSE)
  agg.matrix<-as.data.frame(t(agg.matrix),stringsAsFactors=FALSE)
  colnames(agg.matrix)<-sort(unique(dict.data[,2]))
  return(agg.matrix)
}

request.matrix<-createDocTermMatrix(train.data$request_text_edit_aware)
title.matrix<-createDocTermMatrix(train.data$request_title)

feat.data<-as.data.frame(cbind(train.data[,c('requester_account_age_in_days_at_request',
                              'requester_days_since_first_post_on_raop_at_request',
                              'requester_number_of_comments_at_request',
                              'requester_number_of_comments_in_raop_at_request',
                              'requester_number_of_posts_at_request',
                              'requester_number_of_posts_on_raop_at_request',
                              'requester_number_of_subreddits_at_request',
                              'requester_upvotes_minus_downvotes_at_request',
                              'requester_upvotes_plus_downvotes_at_request',
                              'requester_received_pizza')],
                          substr(as.POSIXct(train.data$unix_timestamp_of_request,origin="1970-01-01"),12,13),
                          weekdays(as.POSIXct(train.data$unix_timestamp_of_request,origin="1970-01-01")),
                          request.matrix[2:nrow(request.matrix),],
                          title.matrix[2:nrow(title.matrix),]))


request.matrix<-createDocTermMatrix(test.data$request_text_edit_aware)
title.matrix<-createDocTermMatrix(test.data$request_title)

val.data<-as.data.frame(cbind(test.data[,c('requester_account_age_in_days_at_request',
                              'requester_days_since_first_post_on_raop_at_request',
                              'requester_number_of_comments_at_request',
                              'requester_number_of_comments_in_raop_at_request',
                              'requester_number_of_posts_at_request',
                              'requester_number_of_posts_on_raop_at_request',
                              'requester_number_of_subreddits_at_request',
                              'requester_upvotes_minus_downvotes_at_request',
                              'requester_upvotes_plus_downvotes_at_request')],
                              substr(as.POSIXct(test.data$unix_timestamp_of_request,origin="1970-01-01"),12,13),
                              weekdays(as.POSIXct(test.data$unix_timestamp_of_request,origin="1970-01-01")),
                  request.matrix[2:nrow(request.matrix),],
                  title.matrix[2:nrow(title.matrix),]))

feat.data[,12:ncol(feat.data)]<-sapply(feat.data[,12:ncol(feat.data)],as.numeric)
val.data[,11:ncol(val.data)]<-sapply(val.data[,11:ncol(val.data)],as.numeric)

feat.data[,11]<-as.factor(feat.data[,11])
val.data[,10]<-as.factor(val.data[,10])
feat.data[,12]<-as.factor(feat.data[,12])
val.data[,11]<-as.factor(val.data[,11])

colnames(feat.data)[11]<-"hour"
colnames(feat.data)[12]<-"dow"

colnames(feat.data)[13:31]<-paste("text.",sort(unique(dict.data[,2])),sep="")
colnames(feat.data)[32:50]<-paste("title.",sort(unique(dict.data[,2])),sep="")

colnames(val.data)[10]<-"hour"
colnames(val.data)[11]<-"dow"

colnames(val.data)[12:30]<-paste("text.",sort(unique(dict.data[,2])),sep="")
colnames(val.data)[31:49]<-paste("title.",sort(unique(dict.data[,2])),sep="")

set.seed(30)
idx.vec<-sample(1:nrow(feat.data),0.8*nrow(feat.data))
train.data<-feat.data[idx.vec,]
test.data<-feat.data[-idx.vec,]

i<-50
j<-1
test.auc<-data.frame(x=1:80,y=1:80)
while(i<=2000)
{
  rf.model<-randomForest(as.factor(train.data$requester_received_pizza)~.,ntree=i,data=train.data)
  rf.pred<-predict(rf.model,test.data,type="prob")
  test.auc[j,]<-data.frame(cbind(i,auc(rf.pred[,2],ifelse(test.data$requester_received_pizza,1,0))))
  i<-i+50
  j<-j+1
}

plot(test.auc[1:40,1],test.auc[1:40,2])
lines(test.auc[1:40,1],test.auc[1:40,2])


rf.model<-randomForest(as.factor(feat.data$requester_received_pizza)~.,ntree=600,data=feat.data,importance=TRUE)
rf.pred<-predict(rf.model,val.data,type="prob")
pred.out<-data.frame(cbind(master.test.data$request_id,rf.pred[,2]))
colnames(pred.out)<-c("request_id", "requester_received_pizza")
write.csv(pred.out,"rf_prediction.csv",row.names=FALSE)