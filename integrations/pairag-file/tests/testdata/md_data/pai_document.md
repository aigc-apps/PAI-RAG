# 2.PAl-Studio/Designer FAQ 2.1. FAQ about algorithm components 

This topic describes the FAQ about algorit hm components.  

What can I do when the format conversion component reports an error? · What can I do when "blob" is displayed on the data present ation page of the PAl? · what canI do when the xl3-auto-arima component reports an error? · What can Ido when the Doc2Vec component reports the CallExecutorToParseTaskFail error?  

What can I do when the format conversion component reports an error? 

By default, the format conversion component runs 100 workers. Make sure that up to 100 data ent ries are converted at a time.  

What can I do when "blob" is displayed on the data presentation page of the PAl?  

Symptom  

On the canvas of the Machine Learning Studio page, when Iright-clicked a component and selected View Data, some text is displayed as "blob."  

Solution  

Characters that cannot be transcoded are displayed as "blob." Ignore this error, because nodes in the downst ream can read and process the data.  

What can I do when the x13-auto-arima component reports an error? 

Make sure that up to 1,200 training data samples are imported into the x13-auto-arima component.  

What can I do when the Doc2Vec component reports the CallExecutorToParseTaskFail error? 

Make sure that the number of data samples imported into the Doc2Vec component is less than $247\,0000\times10000.$ The data is calculated in the formula:  (Number of documents $^+$ Number of  words) x Vector length . Make sure that the number of users who use the Doc2Vec component is less than $42432500\times7712293\times300$ . Exceeding the preceding limits causes memory application failures. If your data size is excessively large, you can reduce the number of data samples and then calculate the formula again. Make sure that the imported data is split into words.  

## 2.2. FAQ about model data 

This topic describes the FAQ about model data.  

Why does my experiment generate an empty model? ·How do I download a model generated in my experiment?  

##### How do I upload data? 

Why does my experiment generate an empty model? 

Symptom  

When I right-clicked a model training component in the canvas and chose Model Option $>$ View Model, the out put is empty.  

Solution  

i. Go to the Machine Learning Studio page. In the left-side navigation pane, click Settings.   
i. Click General. On the General page, select Auto Generate PMML.   
il. Run the experiment again.  

How do I download a model generated in my experiment? 

1. Go to the Machine Learning Studio page. In the left-side navigation pane, click Models.   
2. In the model list, right-clickthe model you want to download and then select Save Model.  

How do I upload data?  

For more information about how to upload a data file, see Prepare data.  

## 2.3. Preview Oss files in Machine Learning Studio 

If you want to preview CsV and JsoN files that are stored in Object Storage Service (Oss) buckets on the Machine Learning Studio platform, you must enable Cross-Origin Resource Sharing (CoRs) for the OsSbuckets.ThistopicdescribeshowtoenableCoRsforOss.  

Limits 

This feature is supported only by Machine Learning Studio 2.0.  

Procedure 

2. On the details page of the Oss bucket, choose Access Cont rol $>$ Cross-Origin Resource Sharing (CORS). In the Cross-Origin Resource Sharing (CORS) section, click Conf igure.  

3. Click Create Rule. In the Create Rule panel, set the following parameters.  

<html><body><table><tr><td>Parameter</td><td>Required</td><td>Description</td></tr><tr><td>Sources</td><td>Yes</td><td>Thesourcesofcross-regionrequeststhatyouwantto allow.Usethefollowingaddressesforthisparameter: https://*.console.aliyun.com http://*.console.aliyun.com</td></tr></table></body></html>
