## PAI-RAG Java Client SDK

This is the java client SDK to call PAI-RAG APIs.

The client code is packaged to [pairag-client-0.1.0.jar](artifacts/pairag-client-0.1.0.jar).

For detailed API usage, please refer to the [API doc](docs/RagQueryApi.md)

### Sample code

```java
package com.mycompany.app;

import com.aliyun.pairag.client.ApiClient;
import com.aliyun.pairag.client.Configuration;
import com.aliyun.pairag.client.model.*;
import com.aliyun.pairag.client.api.RagQueryApi;
import com.aliyun.pairag.client.ApiException;

/**
 * Hello world!
 *
 */
public class App
{
    private static String EasToken = "YOUR_EAS_TOKEN";
    private static String EasEndpoint = "YOUR_EAS_ENDPOINT";

    public static void main( String[] args )
    {
        System.out.println( "Set up client" );
        ApiClient defaultClient = Configuration.getDefaultApiClient();
        defaultClient.setBasePath(EasEndpoint);
        defaultClient.addDefaultHeader("Authorization", EasToken);
        RagQueryApi apiInstance = new RagQueryApi(defaultClient);

        System.out.println( "1. === Rag query example ===" );
        RagQuery ragQuery = new RagQuery(); // RetrievalQuery
        ragQuery.setQuestion("州好玩的地方有哪些？");
        try {
            Object result = apiInstance.aqueryServiceQueryPost(ragQuery);
            System.out.println(result);
        }
        catch (ApiException e)
        {
            System.err.println("Exception when calling RagQueryApi#aqueryServiceQueryPost");
            System.err.println("Status code: " + e.getCode());
            System.err.println("Reason: " + e.getResponseBody());
            System.err.println("Response headers: " + e.getResponseHeaders());
            e.printStackTrace();
        }

        System.out.println( "2. === Retrieval example! ===" );
        RetrievalQuery retrievalQuery = new RetrievalQuery(); // RetrievalQuery
        retrievalQuery.setQuestion("杭州好玩的地方有哪些？");
        try {
            Object result = apiInstance.aqueryRetrievalServiceQueryRetrievalPost(retrievalQuery);
            System.out.println(result);
        }
        catch (ApiException e)
        {
            System.err.println("Exception when calling RagQueryApi#aqueryRetrievalServiceQueryRetrievalPost");
            System.err.println("Status code: " + e.getCode());
            System.err.println("Reason: " + e.getResponseBody());
            System.err.println("Response headers: " + e.getResponseHeaders());
            e.printStackTrace();
        }
    }
}
```
