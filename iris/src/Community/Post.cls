Class Community.Post Extends %Persistent 
{

Parameter DSTIME = "AUTO";

Property Nid As %Integer [ Required ];

Index IDKEY On Nid [ IdKey, Unique ];

Property Type As %String(VALUELIST = ",code_package,documentation,learning_track,video,post");

Property Tags As %String(MAXLEN = 350);

Property Views As %Integer;

Property AvgVote As %Integer;

Property VotesAmount As %Integer;

Property Name As %String(MAXLEN = 250);

Property Created As %TimeStamp;

Property Published As %Boolean;

Property PublishedDate As %TimeStamp;

Property Deleted As %Boolean;

Property Text As %String(MAXLEN = "");

Property Author As %String;

Property CommentsAmount As %Integer;

Property PostType As %String(VALUELIST = ",Question,Article,Announcement,Digest,Discussion");

/// Hash of input row, used to build this object
Property Hash As %String;

Property SubsCount As %Integer;

Property HasCorrectAnswer As %Boolean;

Property FavsCount As %Integer;

Property AcceptedAnswerTS As %TimeStamp;

Property Lang As %String;

Property Translated As %Boolean;

Storage Default
{
<Data name="PostDefaultData">
<Value name="1">
<Value>%%CLASSNAME</Value>
</Value>
<Value name="2">
<Value>Tags</Value>
</Value>
<Value name="3">
<Value>Views</Value>
</Value>
<Value name="4">
<Value>AvgVote</Value>
</Value>
<Value name="5">
<Value>VotesAmount</Value>
</Value>
<Value name="6">
<Value>Name</Value>
</Value>
<Value name="7">
<Value>Created</Value>
</Value>
<Value name="8">
<Value>Text</Value>
</Value>
<Value name="9">
<Value>Link</Value>
</Value>
<Value name="10">
<Value>Author</Value>
</Value>
<Value name="11">
<Value>Type</Value>
</Value>
<Value name="12">
<Value>Published</Value>
</Value>
<Value name="13">
<Value>PublishedDate</Value>
</Value>
<Value name="14">
<Value>Deleted</Value>
</Value>
<Value name="15">
<Value>PostType</Value>
</Value>
<Value name="16">
<Value>Hash</Value>
</Value>
<Value name="17">
<Value>SubsCount</Value>
</Value>
<Value name="18">
<Value>HaveAnswer</Value>
</Value>
<Value name="19">
<Value>HaveCorrectAnswer</Value>
</Value>
<Value name="20">
<Value>HasCorrectAnswer</Value>
</Value>
<Value name="21">
<Value>AcceptedAnswerTS</Value>
</Value>
<Value name="22">
<Value>FavsCount</Value>
</Value>
<Value name="23">
<Value>Language</Value>
</Value>
<Value name="24">
<Value>Lang</Value>
</Value>
<Value name="25">
<Value>Translated</Value>
</Value>
</Data>
<DataLocation>^Community.PostD</DataLocation>
<DefaultData>PostDefaultData</DefaultData>
<IdLocation>^Community.PostD</IdLocation>
<IndexLocation>^Community.PostI</IndexLocation>
<StreamLocation>^Community.PostS</StreamLocation>
<Type>%Storage.Persistent</Type>
}

}

