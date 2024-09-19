# RagQueryApi

All URIs are relative to _http://localhost_

| Method                                                                                                                      | HTTP request                           | Description              |
| --------------------------------------------------------------------------------------------------------------------------- | -------------------------------------- | ------------------------ |
| [**aconfigServiceConfigGet**](RagQueryApi.md#aconfigServiceConfigGet)                                                       | **GET** /service/config                | Aconfig                  |
| [**aloadAgentConfigServiceConfigAgentPost**](RagQueryApi.md#aloadAgentConfigServiceConfigAgentPost)                         | **POST** /service/config/agent         | Aload Agent Config       |
| [**aqueryAgentServiceQueryAgentPost**](RagQueryApi.md#aqueryAgentServiceQueryAgentPost)                                     | **POST** /service/query/agent          | Aquery Agent             |
| [**aqueryAnalysisServiceQueryDataAnalysisPost**](RagQueryApi.md#aqueryAnalysisServiceQueryDataAnalysisPost)                 | **POST** /service/query/data_analysis  | Aquery Analysis          |
| [**aqueryLlmServiceQueryLlmPost**](RagQueryApi.md#aqueryLlmServiceQueryLlmPost)                                             | **POST** /service/query/llm            | Aquery Llm               |
| [**aqueryRetrievalServiceQueryRetrievalPost**](RagQueryApi.md#aqueryRetrievalServiceQueryRetrievalPost)                     | **POST** /service/query/retrieval      | Aquery Retrieval         |
| [**aquerySearchServiceQuerySearchPost**](RagQueryApi.md#aquerySearchServiceQuerySearchPost)                                 | **POST** /service/query/search         | Aquery Search            |
| [**aqueryServiceQueryPost**](RagQueryApi.md#aqueryServiceQueryPost)                                                         | **POST** /service/query                | Aquery                   |
| [**aupdateServiceConfigPatch**](RagQueryApi.md#aupdateServiceConfigPatch)                                                   | **PATCH** /service/config              | Aupdate                  |
| [**batchEvaluateServiceEvaluatePost**](RagQueryApi.md#batchEvaluateServiceEvaluatePost)                                     | **POST** /service/evaluate             | Batch Evaluate           |
| [**batchResponseEvaluateServiceEvaluateResponsePost**](RagQueryApi.md#batchResponseEvaluateServiceEvaluateResponsePost)     | **POST** /service/evaluate/response    | Batch Response Evaluate  |
| [**batchRetrievalEvaluateServiceEvaluateRetrievalPost**](RagQueryApi.md#batchRetrievalEvaluateServiceEvaluateRetrievalPost) | **POST** /service/evaluate/retrieval   | Batch Retrieval Evaluate |
| [**generateQaDatasetServiceEvaluateGeneratePost**](RagQueryApi.md#generateQaDatasetServiceEvaluateGeneratePost)             | **POST** /service/evaluate/generate    | Generate Qa Dataset      |
| [**taskStatusServiceGetUploadStateGet**](RagQueryApi.md#taskStatusServiceGetUploadStateGet)                                 | **GET** /service/get_upload_state      | Task Status              |
| [**uploadDataServiceUploadDataPost**](RagQueryApi.md#uploadDataServiceUploadDataPost)                                       | **POST** /service/upload_data          | Upload Data              |
| [**uploadDatasheetServiceUploadDatasheetPost**](RagQueryApi.md#uploadDatasheetServiceUploadDatasheetPost)                   | **POST** /service/upload_datasheet     | Upload Datasheet         |
| [**uploadOssDataServiceUploadDataFromOssPost**](RagQueryApi.md#uploadOssDataServiceUploadDataFromOssPost)                   | **POST** /service/upload_data_from_oss | Upload Oss Data          |

<a id="aconfigServiceConfigGet"></a>

# **aconfigServiceConfigGet**

> Object aconfigServiceConfigGet()

Aconfig

### Example

```java
// Import classes:
import com.aliyun.pairag.client.ApiClient;
import com.aliyun.pairag.client.ApiException;
import com.aliyun.pairag.client.Configuration;
import com.aliyun.pairag.client.models.*;
import com.aliyun.pairag.client.api.RagQueryApi;

public class Example {
  public static void main(String[] args) {
    ApiClient defaultClient = Configuration.getDefaultApiClient();
    defaultClient.setBasePath("http://localhost");

    RagQueryApi apiInstance = new RagQueryApi(defaultClient);
    try {
      Object result = apiInstance.aconfigServiceConfigGet();
      System.out.println(result);
    } catch (ApiException e) {
      System.err.println("Exception when calling RagQueryApi#aconfigServiceConfigGet");
      System.err.println("Status code: " + e.getCode());
      System.err.println("Reason: " + e.getResponseBody());
      System.err.println("Response headers: " + e.getResponseHeaders());
      e.printStackTrace();
    }
  }
}
```

### Parameters

This endpoint does not need any parameter.

### Return type

**Object**

### Authorization

No authorization required

### HTTP request headers

- **Content-Type**: Not defined
- **Accept**: application/json

### HTTP response details

| Status code | Description         | Response headers |
| ----------- | ------------------- | ---------------- |
| **200**     | Successful Response | -                |

<a id="aloadAgentConfigServiceConfigAgentPost"></a>

# **aloadAgentConfigServiceConfigAgentPost**

> Object aloadAgentConfigServiceConfigAgentPost(\_file)

Aload Agent Config

### Example

```java
// Import classes:
import com.aliyun.pairag.client.ApiClient;
import com.aliyun.pairag.client.ApiException;
import com.aliyun.pairag.client.Configuration;
import com.aliyun.pairag.client.models.*;
import com.aliyun.pairag.client.api.RagQueryApi;

public class Example {
  public static void main(String[] args) {
    ApiClient defaultClient = Configuration.getDefaultApiClient();
    defaultClient.setBasePath("http://localhost");

    RagQueryApi apiInstance = new RagQueryApi(defaultClient);
    File _file = new File("/path/to/file"); // File |
    try {
      Object result = apiInstance.aloadAgentConfigServiceConfigAgentPost(_file);
      System.out.println(result);
    } catch (ApiException e) {
      System.err.println("Exception when calling RagQueryApi#aloadAgentConfigServiceConfigAgentPost");
      System.err.println("Status code: " + e.getCode());
      System.err.println("Reason: " + e.getResponseBody());
      System.err.println("Response headers: " + e.getResponseHeaders());
      e.printStackTrace();
    }
  }
}
```

### Parameters

| Name       | Type     | Description | Notes |
| ---------- | -------- | ----------- | ----- |
| **\_file** | **File** |             |       |

### Return type

**Object**

### Authorization

No authorization required

### HTTP request headers

- **Content-Type**: multipart/form-data
- **Accept**: application/json

### HTTP response details

| Status code | Description         | Response headers |
| ----------- | ------------------- | ---------------- |
| **200**     | Successful Response | -                |
| **422**     | Validation Error    | -                |

<a id="aqueryAgentServiceQueryAgentPost"></a>

# **aqueryAgentServiceQueryAgentPost**

> LlmResponse aqueryAgentServiceQueryAgentPost(ragQuery)

Aquery Agent

### Example

```java
// Import classes:
import com.aliyun.pairag.client.ApiClient;
import com.aliyun.pairag.client.ApiException;
import com.aliyun.pairag.client.Configuration;
import com.aliyun.pairag.client.models.*;
import com.aliyun.pairag.client.api.RagQueryApi;

public class Example {
  public static void main(String[] args) {
    ApiClient defaultClient = Configuration.getDefaultApiClient();
    defaultClient.setBasePath("http://localhost");

    RagQueryApi apiInstance = new RagQueryApi(defaultClient);
    RagQuery ragQuery = new RagQuery(); // RagQuery |
    try {
      LlmResponse result = apiInstance.aqueryAgentServiceQueryAgentPost(ragQuery);
      System.out.println(result);
    } catch (ApiException e) {
      System.err.println("Exception when calling RagQueryApi#aqueryAgentServiceQueryAgentPost");
      System.err.println("Status code: " + e.getCode());
      System.err.println("Reason: " + e.getResponseBody());
      System.err.println("Response headers: " + e.getResponseHeaders());
      e.printStackTrace();
    }
  }
}
```

### Parameters

| Name         | Type                        | Description | Notes |
| ------------ | --------------------------- | ----------- | ----- |
| **ragQuery** | [**RagQuery**](RagQuery.md) |             |       |

### Return type

[**LlmResponse**](LlmResponse.md)

### Authorization

No authorization required

### HTTP request headers

- **Content-Type**: application/json
- **Accept**: application/json

### HTTP response details

| Status code | Description         | Response headers |
| ----------- | ------------------- | ---------------- |
| **200**     | Successful Response | -                |
| **422**     | Validation Error    | -                |

<a id="aqueryAnalysisServiceQueryDataAnalysisPost"></a>

# **aqueryAnalysisServiceQueryDataAnalysisPost**

> Object aqueryAnalysisServiceQueryDataAnalysisPost(ragQuery)

Aquery Analysis

### Example

```java
// Import classes:
import com.aliyun.pairag.client.ApiClient;
import com.aliyun.pairag.client.ApiException;
import com.aliyun.pairag.client.Configuration;
import com.aliyun.pairag.client.models.*;
import com.aliyun.pairag.client.api.RagQueryApi;

public class Example {
  public static void main(String[] args) {
    ApiClient defaultClient = Configuration.getDefaultApiClient();
    defaultClient.setBasePath("http://localhost");

    RagQueryApi apiInstance = new RagQueryApi(defaultClient);
    RagQuery ragQuery = new RagQuery(); // RagQuery |
    try {
      Object result = apiInstance.aqueryAnalysisServiceQueryDataAnalysisPost(ragQuery);
      System.out.println(result);
    } catch (ApiException e) {
      System.err.println("Exception when calling RagQueryApi#aqueryAnalysisServiceQueryDataAnalysisPost");
      System.err.println("Status code: " + e.getCode());
      System.err.println("Reason: " + e.getResponseBody());
      System.err.println("Response headers: " + e.getResponseHeaders());
      e.printStackTrace();
    }
  }
}
```

### Parameters

| Name         | Type                        | Description | Notes |
| ------------ | --------------------------- | ----------- | ----- |
| **ragQuery** | [**RagQuery**](RagQuery.md) |             |       |

### Return type

**Object**

### Authorization

No authorization required

### HTTP request headers

- **Content-Type**: application/json
- **Accept**: application/json

### HTTP response details

| Status code | Description         | Response headers |
| ----------- | ------------------- | ---------------- |
| **200**     | Successful Response | -                |
| **422**     | Validation Error    | -                |

<a id="aqueryLlmServiceQueryLlmPost"></a>

# **aqueryLlmServiceQueryLlmPost**

> Object aqueryLlmServiceQueryLlmPost(ragQuery)

Aquery Llm

### Example

```java
// Import classes:
import com.aliyun.pairag.client.ApiClient;
import com.aliyun.pairag.client.ApiException;
import com.aliyun.pairag.client.Configuration;
import com.aliyun.pairag.client.models.*;
import com.aliyun.pairag.client.api.RagQueryApi;

public class Example {
  public static void main(String[] args) {
    ApiClient defaultClient = Configuration.getDefaultApiClient();
    defaultClient.setBasePath("http://localhost");

    RagQueryApi apiInstance = new RagQueryApi(defaultClient);
    RagQuery ragQuery = new RagQuery(); // RagQuery |
    try {
      Object result = apiInstance.aqueryLlmServiceQueryLlmPost(ragQuery);
      System.out.println(result);
    } catch (ApiException e) {
      System.err.println("Exception when calling RagQueryApi#aqueryLlmServiceQueryLlmPost");
      System.err.println("Status code: " + e.getCode());
      System.err.println("Reason: " + e.getResponseBody());
      System.err.println("Response headers: " + e.getResponseHeaders());
      e.printStackTrace();
    }
  }
}
```

### Parameters

| Name         | Type                        | Description | Notes |
| ------------ | --------------------------- | ----------- | ----- |
| **ragQuery** | [**RagQuery**](RagQuery.md) |             |       |

### Return type

**Object**

### Authorization

No authorization required

### HTTP request headers

- **Content-Type**: application/json
- **Accept**: application/json

### HTTP response details

| Status code | Description         | Response headers |
| ----------- | ------------------- | ---------------- |
| **200**     | Successful Response | -                |
| **422**     | Validation Error    | -                |

<a id="aqueryRetrievalServiceQueryRetrievalPost"></a>

# **aqueryRetrievalServiceQueryRetrievalPost**

> Object aqueryRetrievalServiceQueryRetrievalPost(retrievalQuery)

Aquery Retrieval

### Example

```java
// Import classes:
import com.aliyun.pairag.client.ApiClient;
import com.aliyun.pairag.client.ApiException;
import com.aliyun.pairag.client.Configuration;
import com.aliyun.pairag.client.models.*;
import com.aliyun.pairag.client.api.RagQueryApi;

public class Example {
  public static void main(String[] args) {
    ApiClient defaultClient = Configuration.getDefaultApiClient();
    defaultClient.setBasePath("http://localhost");

    RagQueryApi apiInstance = new RagQueryApi(defaultClient);
    RetrievalQuery retrievalQuery = new RetrievalQuery(); // RetrievalQuery |
    try {
      Object result = apiInstance.aqueryRetrievalServiceQueryRetrievalPost(retrievalQuery);
      System.out.println(result);
    } catch (ApiException e) {
      System.err.println("Exception when calling RagQueryApi#aqueryRetrievalServiceQueryRetrievalPost");
      System.err.println("Status code: " + e.getCode());
      System.err.println("Reason: " + e.getResponseBody());
      System.err.println("Response headers: " + e.getResponseHeaders());
      e.printStackTrace();
    }
  }
}
```

### Parameters

| Name               | Type                                    | Description | Notes |
| ------------------ | --------------------------------------- | ----------- | ----- |
| **retrievalQuery** | [**RetrievalQuery**](RetrievalQuery.md) |             |       |

### Return type

**Object**

### Authorization

No authorization required

### HTTP request headers

- **Content-Type**: application/json
- **Accept**: application/json

### HTTP response details

| Status code | Description         | Response headers |
| ----------- | ------------------- | ---------------- |
| **200**     | Successful Response | -                |
| **422**     | Validation Error    | -                |

<a id="aquerySearchServiceQuerySearchPost"></a>

# **aquerySearchServiceQuerySearchPost**

> Object aquerySearchServiceQuerySearchPost(ragQuery)

Aquery Search

### Example

```java
// Import classes:
import com.aliyun.pairag.client.ApiClient;
import com.aliyun.pairag.client.ApiException;
import com.aliyun.pairag.client.Configuration;
import com.aliyun.pairag.client.models.*;
import com.aliyun.pairag.client.api.RagQueryApi;

public class Example {
  public static void main(String[] args) {
    ApiClient defaultClient = Configuration.getDefaultApiClient();
    defaultClient.setBasePath("http://localhost");

    RagQueryApi apiInstance = new RagQueryApi(defaultClient);
    RagQuery ragQuery = new RagQuery(); // RagQuery |
    try {
      Object result = apiInstance.aquerySearchServiceQuerySearchPost(ragQuery);
      System.out.println(result);
    } catch (ApiException e) {
      System.err.println("Exception when calling RagQueryApi#aquerySearchServiceQuerySearchPost");
      System.err.println("Status code: " + e.getCode());
      System.err.println("Reason: " + e.getResponseBody());
      System.err.println("Response headers: " + e.getResponseHeaders());
      e.printStackTrace();
    }
  }
}
```

### Parameters

| Name         | Type                        | Description | Notes |
| ------------ | --------------------------- | ----------- | ----- |
| **ragQuery** | [**RagQuery**](RagQuery.md) |             |       |

### Return type

**Object**

### Authorization

No authorization required

### HTTP request headers

- **Content-Type**: application/json
- **Accept**: application/json

### HTTP response details

| Status code | Description         | Response headers |
| ----------- | ------------------- | ---------------- |
| **200**     | Successful Response | -                |
| **422**     | Validation Error    | -                |

<a id="aqueryServiceQueryPost"></a>

# **aqueryServiceQueryPost**

> Object aqueryServiceQueryPost(ragQuery)

Aquery

### Example

```java
// Import classes:
import com.aliyun.pairag.client.ApiClient;
import com.aliyun.pairag.client.ApiException;
import com.aliyun.pairag.client.Configuration;
import com.aliyun.pairag.client.models.*;
import com.aliyun.pairag.client.api.RagQueryApi;

public class Example {
  public static void main(String[] args) {
    ApiClient defaultClient = Configuration.getDefaultApiClient();
    defaultClient.setBasePath("http://localhost");

    RagQueryApi apiInstance = new RagQueryApi(defaultClient);
    RagQuery ragQuery = new RagQuery(); // RagQuery |
    try {
      Object result = apiInstance.aqueryServiceQueryPost(ragQuery);
      System.out.println(result);
    } catch (ApiException e) {
      System.err.println("Exception when calling RagQueryApi#aqueryServiceQueryPost");
      System.err.println("Status code: " + e.getCode());
      System.err.println("Reason: " + e.getResponseBody());
      System.err.println("Response headers: " + e.getResponseHeaders());
      e.printStackTrace();
    }
  }
}
```

### Parameters

| Name         | Type                        | Description | Notes |
| ------------ | --------------------------- | ----------- | ----- |
| **ragQuery** | [**RagQuery**](RagQuery.md) |             |       |

### Return type

**Object**

### Authorization

No authorization required

### HTTP request headers

- **Content-Type**: application/json
- **Accept**: application/json

### HTTP response details

| Status code | Description         | Response headers |
| ----------- | ------------------- | ---------------- |
| **200**     | Successful Response | -                |
| **422**     | Validation Error    | -                |

<a id="aupdateServiceConfigPatch"></a>

# **aupdateServiceConfigPatch**

> Object aupdateServiceConfigPatch(body)

Aupdate

### Example

```java
// Import classes:
import com.aliyun.pairag.client.ApiClient;
import com.aliyun.pairag.client.ApiException;
import com.aliyun.pairag.client.Configuration;
import com.aliyun.pairag.client.models.*;
import com.aliyun.pairag.client.api.RagQueryApi;

public class Example {
  public static void main(String[] args) {
    ApiClient defaultClient = Configuration.getDefaultApiClient();
    defaultClient.setBasePath("http://localhost");

    RagQueryApi apiInstance = new RagQueryApi(defaultClient);
    Object body = null; // Object |
    try {
      Object result = apiInstance.aupdateServiceConfigPatch(body);
      System.out.println(result);
    } catch (ApiException e) {
      System.err.println("Exception when calling RagQueryApi#aupdateServiceConfigPatch");
      System.err.println("Status code: " + e.getCode());
      System.err.println("Reason: " + e.getResponseBody());
      System.err.println("Response headers: " + e.getResponseHeaders());
      e.printStackTrace();
    }
  }
}
```

### Parameters

| Name     | Type       | Description | Notes      |
| -------- | ---------- | ----------- | ---------- |
| **body** | **Object** |             | [optional] |

### Return type

**Object**

### Authorization

No authorization required

### HTTP request headers

- **Content-Type**: application/json
- **Accept**: application/json

### HTTP response details

| Status code | Description         | Response headers |
| ----------- | ------------------- | ---------------- |
| **200**     | Successful Response | -                |
| **422**     | Validation Error    | -                |

<a id="batchEvaluateServiceEvaluatePost"></a>

# **batchEvaluateServiceEvaluatePost**

> Object batchEvaluateServiceEvaluatePost(overwrite)

Batch Evaluate

### Example

```java
// Import classes:
import com.aliyun.pairag.client.ApiClient;
import com.aliyun.pairag.client.ApiException;
import com.aliyun.pairag.client.Configuration;
import com.aliyun.pairag.client.models.*;
import com.aliyun.pairag.client.api.RagQueryApi;

public class Example {
  public static void main(String[] args) {
    ApiClient defaultClient = Configuration.getDefaultApiClient();
    defaultClient.setBasePath("http://localhost");

    RagQueryApi apiInstance = new RagQueryApi(defaultClient);
    Boolean overwrite = false; // Boolean |
    try {
      Object result = apiInstance.batchEvaluateServiceEvaluatePost(overwrite);
      System.out.println(result);
    } catch (ApiException e) {
      System.err.println("Exception when calling RagQueryApi#batchEvaluateServiceEvaluatePost");
      System.err.println("Status code: " + e.getCode());
      System.err.println("Reason: " + e.getResponseBody());
      System.err.println("Response headers: " + e.getResponseHeaders());
      e.printStackTrace();
    }
  }
}
```

### Parameters

| Name          | Type        | Description | Notes                         |
| ------------- | ----------- | ----------- | ----------------------------- |
| **overwrite** | **Boolean** |             | [optional] [default to false] |

### Return type

**Object**

### Authorization

No authorization required

### HTTP request headers

- **Content-Type**: Not defined
- **Accept**: application/json

### HTTP response details

| Status code | Description         | Response headers |
| ----------- | ------------------- | ---------------- |
| **200**     | Successful Response | -                |
| **422**     | Validation Error    | -                |

<a id="batchResponseEvaluateServiceEvaluateResponsePost"></a>

# **batchResponseEvaluateServiceEvaluateResponsePost**

> Object batchResponseEvaluateServiceEvaluateResponsePost(overwrite)

Batch Response Evaluate

### Example

```java
// Import classes:
import com.aliyun.pairag.client.ApiClient;
import com.aliyun.pairag.client.ApiException;
import com.aliyun.pairag.client.Configuration;
import com.aliyun.pairag.client.models.*;
import com.aliyun.pairag.client.api.RagQueryApi;

public class Example {
  public static void main(String[] args) {
    ApiClient defaultClient = Configuration.getDefaultApiClient();
    defaultClient.setBasePath("http://localhost");

    RagQueryApi apiInstance = new RagQueryApi(defaultClient);
    Boolean overwrite = false; // Boolean |
    try {
      Object result = apiInstance.batchResponseEvaluateServiceEvaluateResponsePost(overwrite);
      System.out.println(result);
    } catch (ApiException e) {
      System.err.println("Exception when calling RagQueryApi#batchResponseEvaluateServiceEvaluateResponsePost");
      System.err.println("Status code: " + e.getCode());
      System.err.println("Reason: " + e.getResponseBody());
      System.err.println("Response headers: " + e.getResponseHeaders());
      e.printStackTrace();
    }
  }
}
```

### Parameters

| Name          | Type        | Description | Notes                         |
| ------------- | ----------- | ----------- | ----------------------------- |
| **overwrite** | **Boolean** |             | [optional] [default to false] |

### Return type

**Object**

### Authorization

No authorization required

### HTTP request headers

- **Content-Type**: Not defined
- **Accept**: application/json

### HTTP response details

| Status code | Description         | Response headers |
| ----------- | ------------------- | ---------------- |
| **200**     | Successful Response | -                |
| **422**     | Validation Error    | -                |

<a id="batchRetrievalEvaluateServiceEvaluateRetrievalPost"></a>

# **batchRetrievalEvaluateServiceEvaluateRetrievalPost**

> Object batchRetrievalEvaluateServiceEvaluateRetrievalPost(overwrite)

Batch Retrieval Evaluate

### Example

```java
// Import classes:
import com.aliyun.pairag.client.ApiClient;
import com.aliyun.pairag.client.ApiException;
import com.aliyun.pairag.client.Configuration;
import com.aliyun.pairag.client.models.*;
import com.aliyun.pairag.client.api.RagQueryApi;

public class Example {
  public static void main(String[] args) {
    ApiClient defaultClient = Configuration.getDefaultApiClient();
    defaultClient.setBasePath("http://localhost");

    RagQueryApi apiInstance = new RagQueryApi(defaultClient);
    Boolean overwrite = false; // Boolean |
    try {
      Object result = apiInstance.batchRetrievalEvaluateServiceEvaluateRetrievalPost(overwrite);
      System.out.println(result);
    } catch (ApiException e) {
      System.err.println("Exception when calling RagQueryApi#batchRetrievalEvaluateServiceEvaluateRetrievalPost");
      System.err.println("Status code: " + e.getCode());
      System.err.println("Reason: " + e.getResponseBody());
      System.err.println("Response headers: " + e.getResponseHeaders());
      e.printStackTrace();
    }
  }
}
```

### Parameters

| Name          | Type        | Description | Notes                         |
| ------------- | ----------- | ----------- | ----------------------------- |
| **overwrite** | **Boolean** |             | [optional] [default to false] |

### Return type

**Object**

### Authorization

No authorization required

### HTTP request headers

- **Content-Type**: Not defined
- **Accept**: application/json

### HTTP response details

| Status code | Description         | Response headers |
| ----------- | ------------------- | ---------------- |
| **200**     | Successful Response | -                |
| **422**     | Validation Error    | -                |

<a id="generateQaDatasetServiceEvaluateGeneratePost"></a>

# **generateQaDatasetServiceEvaluateGeneratePost**

> Object generateQaDatasetServiceEvaluateGeneratePost(overwrite)

Generate Qa Dataset

### Example

```java
// Import classes:
import com.aliyun.pairag.client.ApiClient;
import com.aliyun.pairag.client.ApiException;
import com.aliyun.pairag.client.Configuration;
import com.aliyun.pairag.client.models.*;
import com.aliyun.pairag.client.api.RagQueryApi;

public class Example {
  public static void main(String[] args) {
    ApiClient defaultClient = Configuration.getDefaultApiClient();
    defaultClient.setBasePath("http://localhost");

    RagQueryApi apiInstance = new RagQueryApi(defaultClient);
    Boolean overwrite = false; // Boolean |
    try {
      Object result = apiInstance.generateQaDatasetServiceEvaluateGeneratePost(overwrite);
      System.out.println(result);
    } catch (ApiException e) {
      System.err.println("Exception when calling RagQueryApi#generateQaDatasetServiceEvaluateGeneratePost");
      System.err.println("Status code: " + e.getCode());
      System.err.println("Reason: " + e.getResponseBody());
      System.err.println("Response headers: " + e.getResponseHeaders());
      e.printStackTrace();
    }
  }
}
```

### Parameters

| Name          | Type        | Description | Notes                         |
| ------------- | ----------- | ----------- | ----------------------------- |
| **overwrite** | **Boolean** |             | [optional] [default to false] |

### Return type

**Object**

### Authorization

No authorization required

### HTTP request headers

- **Content-Type**: Not defined
- **Accept**: application/json

### HTTP response details

| Status code | Description         | Response headers |
| ----------- | ------------------- | ---------------- |
| **200**     | Successful Response | -                |
| **422**     | Validation Error    | -                |

<a id="taskStatusServiceGetUploadStateGet"></a>

# **taskStatusServiceGetUploadStateGet**

> Object taskStatusServiceGetUploadStateGet(taskId)

Task Status

### Example

```java
// Import classes:
import com.aliyun.pairag.client.ApiClient;
import com.aliyun.pairag.client.ApiException;
import com.aliyun.pairag.client.Configuration;
import com.aliyun.pairag.client.models.*;
import com.aliyun.pairag.client.api.RagQueryApi;

public class Example {
  public static void main(String[] args) {
    ApiClient defaultClient = Configuration.getDefaultApiClient();
    defaultClient.setBasePath("http://localhost");

    RagQueryApi apiInstance = new RagQueryApi(defaultClient);
    String taskId = "taskId_example"; // String |
    try {
      Object result = apiInstance.taskStatusServiceGetUploadStateGet(taskId);
      System.out.println(result);
    } catch (ApiException e) {
      System.err.println("Exception when calling RagQueryApi#taskStatusServiceGetUploadStateGet");
      System.err.println("Status code: " + e.getCode());
      System.err.println("Reason: " + e.getResponseBody());
      System.err.println("Response headers: " + e.getResponseHeaders());
      e.printStackTrace();
    }
  }
}
```

### Parameters

| Name       | Type       | Description | Notes |
| ---------- | ---------- | ----------- | ----- |
| **taskId** | **String** |             |       |

### Return type

**Object**

### Authorization

No authorization required

### HTTP request headers

- **Content-Type**: Not defined
- **Accept**: application/json

### HTTP response details

| Status code | Description         | Response headers |
| ----------- | ------------------- | ---------------- |
| **200**     | Successful Response | -                |
| **422**     | Validation Error    | -                |

<a id="uploadDataServiceUploadDataPost"></a>

# **uploadDataServiceUploadDataPost**

> Object uploadDataServiceUploadDataPost(files, faissPath, enableRaptor)

Upload Data

### Example

```java
// Import classes:
import com.aliyun.pairag.client.ApiClient;
import com.aliyun.pairag.client.ApiException;
import com.aliyun.pairag.client.Configuration;
import com.aliyun.pairag.client.models.*;
import com.aliyun.pairag.client.api.RagQueryApi;

public class Example {
  public static void main(String[] args) {
    ApiClient defaultClient = Configuration.getDefaultApiClient();
    defaultClient.setBasePath("http://localhost");

    RagQueryApi apiInstance = new RagQueryApi(defaultClient);
    List<File> files = Arrays.asList(); // List<File> |
    String faissPath = "faissPath_example"; // String |
    Boolean enableRaptor = false; // Boolean |
    try {
      Object result = apiInstance.uploadDataServiceUploadDataPost(files, faissPath, enableRaptor);
      System.out.println(result);
    } catch (ApiException e) {
      System.err.println("Exception when calling RagQueryApi#uploadDataServiceUploadDataPost");
      System.err.println("Status code: " + e.getCode());
      System.err.println("Reason: " + e.getResponseBody());
      System.err.println("Response headers: " + e.getResponseHeaders());
      e.printStackTrace();
    }
  }
}
```

### Parameters

| Name             | Type                 | Description | Notes                         |
| ---------------- | -------------------- | ----------- | ----------------------------- |
| **files**        | **List&lt;File&gt;** |             |                               |
| **faissPath**    | **String**           |             | [optional]                    |
| **enableRaptor** | **Boolean**          |             | [optional] [default to false] |

### Return type

**Object**

### Authorization

No authorization required

### HTTP request headers

- **Content-Type**: multipart/form-data
- **Accept**: application/json

### HTTP response details

| Status code | Description         | Response headers |
| ----------- | ------------------- | ---------------- |
| **200**     | Successful Response | -                |
| **422**     | Validation Error    | -                |

<a id="uploadDatasheetServiceUploadDatasheetPost"></a>

# **uploadDatasheetServiceUploadDatasheetPost**

> Object uploadDatasheetServiceUploadDatasheetPost(\_file)

Upload Datasheet

### Example

```java
// Import classes:
import com.aliyun.pairag.client.ApiClient;
import com.aliyun.pairag.client.ApiException;
import com.aliyun.pairag.client.Configuration;
import com.aliyun.pairag.client.models.*;
import com.aliyun.pairag.client.api.RagQueryApi;

public class Example {
  public static void main(String[] args) {
    ApiClient defaultClient = Configuration.getDefaultApiClient();
    defaultClient.setBasePath("http://localhost");

    RagQueryApi apiInstance = new RagQueryApi(defaultClient);
    File _file = new File("/path/to/file"); // File |
    try {
      Object result = apiInstance.uploadDatasheetServiceUploadDatasheetPost(_file);
      System.out.println(result);
    } catch (ApiException e) {
      System.err.println("Exception when calling RagQueryApi#uploadDatasheetServiceUploadDatasheetPost");
      System.err.println("Status code: " + e.getCode());
      System.err.println("Reason: " + e.getResponseBody());
      System.err.println("Response headers: " + e.getResponseHeaders());
      e.printStackTrace();
    }
  }
}
```

### Parameters

| Name       | Type     | Description | Notes |
| ---------- | -------- | ----------- | ----- |
| **\_file** | **File** |             |       |

### Return type

**Object**

### Authorization

No authorization required

### HTTP request headers

- **Content-Type**: multipart/form-data
- **Accept**: application/json

### HTTP response details

| Status code | Description         | Response headers |
| ----------- | ------------------- | ---------------- |
| **200**     | Successful Response | -                |
| **422**     | Validation Error    | -                |

<a id="uploadOssDataServiceUploadDataFromOssPost"></a>

# **uploadOssDataServiceUploadDataFromOssPost**

> Object uploadOssDataServiceUploadDataFromOssPost(ossPrefix, faissPath, enableRaptor)

Upload Oss Data

### Example

```java
// Import classes:
import com.aliyun.pairag.client.ApiClient;
import com.aliyun.pairag.client.ApiException;
import com.aliyun.pairag.client.Configuration;
import com.aliyun.pairag.client.models.*;
import com.aliyun.pairag.client.api.RagQueryApi;

public class Example {
  public static void main(String[] args) {
    ApiClient defaultClient = Configuration.getDefaultApiClient();
    defaultClient.setBasePath("http://localhost");

    RagQueryApi apiInstance = new RagQueryApi(defaultClient);
    String ossPrefix = "ossPrefix_example"; // String |
    String faissPath = "faissPath_example"; // String |
    Boolean enableRaptor = false; // Boolean |
    try {
      Object result = apiInstance.uploadOssDataServiceUploadDataFromOssPost(ossPrefix, faissPath, enableRaptor);
      System.out.println(result);
    } catch (ApiException e) {
      System.err.println("Exception when calling RagQueryApi#uploadOssDataServiceUploadDataFromOssPost");
      System.err.println("Status code: " + e.getCode());
      System.err.println("Reason: " + e.getResponseBody());
      System.err.println("Response headers: " + e.getResponseHeaders());
      e.printStackTrace();
    }
  }
}
```

### Parameters

| Name             | Type        | Description | Notes                         |
| ---------------- | ----------- | ----------- | ----------------------------- |
| **ossPrefix**    | **String**  |             | [optional]                    |
| **faissPath**    | **String**  |             | [optional]                    |
| **enableRaptor** | **Boolean** |             | [optional] [default to false] |

### Return type

**Object**

### Authorization

No authorization required

### HTTP request headers

- **Content-Type**: Not defined
- **Accept**: application/json

### HTTP response details

| Status code | Description         | Response headers |
| ----------- | ------------------- | ---------------- |
| **200**     | Successful Response | -                |
| **422**     | Validation Error    | -                |
