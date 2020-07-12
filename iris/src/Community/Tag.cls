Class Community.Tag Extends %Persistent
{

Property Name As %String(MAXLEN = "");

Index NameIdx On Name [ IdKey, Unique ];

Property Description As %String(MAXLEN = "");

Storage Default
{
<Data name="TagDefaultData">
<Value name="1">
<Value>%%CLASSNAME</Value>
</Value>
<Value name="2">
<Value>Name</Value>
</Value>
<Value name="3">
<Value>Description</Value>
</Value>
</Data>
<DataLocation>^Community.TagD</DataLocation>
<DefaultData>TagsDefaultData</DefaultData>
<IdLocation>^Community.TagD</IdLocation>
<IndexLocation>^Community.TagI</IndexLocation>
<StreamLocation>^Community.TagS</StreamLocation>
<Type>%Storage.Persistent</Type>
}

}
