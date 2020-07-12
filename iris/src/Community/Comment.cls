Class Community.Comment Extends %Persistent
{

Parameter DSTIME = "AUTO";

Property Id As %Integer [ Required ];

Index IDKEY On (Type, Id) [ IdKey, Unique ];

Property AvgVote As %Integer;

Property VotesAmount As %Integer;

Property Post As %String;

Index PostIndex On Post;

Property Author As %String;

Index AuthorIndex On Author;

Property Created As %TimeStamp;

Property Text As %String(MAXLEN = "");

Property Type As %String(VALUELIST = ",answer,comment");

Property Correct As %Boolean;

/// Hash of input row, used to build this object
Property Hash As %String;

Property TextHash As %String;

Index TextHashIndex On TextHash;

Property AcceptedAnswerTS As %TimeStamp;

Storage Default
{
<Data name="CommentDefaultData">
<Value name="1">
<Value>%%CLASSNAME</Value>
</Value>
<Value name="2">
<Value>AvgVote</Value>
</Value>
<Value name="3">
<Value>VotesAmount</Value>
</Value>
<Value name="4">
<Value>Post</Value>
</Value>
<Value name="5">
<Value>Author</Value>
</Value>
<Value name="6">
<Value>Link</Value>
</Value>
<Value name="7">
<Value>Created</Value>
</Value>
<Value name="8">
<Value>Text</Value>
</Value>
<Value name="9">
<Value>Deleted</Value>
</Value>
<Value name="10">
<Value>Hash</Value>
</Value>
<Value name="11">
<Value>Correct</Value>
</Value>
<Value name="12">
<Value>TextHash</Value>
</Value>
<Value name="13">
<Value>AsseptedAnswerTS</Value>
</Value>
<Value name="14">
<Value>AcceptedAnswerTS</Value>
</Value>
<Value name="15">
<Value>Language</Value>
</Value>
<Value name="16">
<Value>Lang</Value>
</Value>
</Data>
<DataLocation>^Community.CommentD</DataLocation>
<DefaultData>CommentDefaultData</DefaultData>
<IdLocation>^Community.CommentD</IdLocation>
<IndexLocation>^Community.CommentI</IndexLocation>
<StreamLocation>^Community.CommentS</StreamLocation>
<Type>%Storage.Persistent</Type>
}

}
