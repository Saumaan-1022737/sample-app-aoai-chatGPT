{
    "definition": {
        "$schema": "https://schema.management.azure.com/providers/Microsoft.Logic/schemas/2016-06-01/workflowdefinition.json#",
        "actions": {
            "Execute_JavaScript_Code": {
                "type": "JavaScriptCode",
                "inputs": {
                    "code": "var emailBody = workflowContext.trigger.outputs.body.body;\r\n\r\n// Remove all HTML tags\r\nvar cleanEmail = emailBody.replace(/<\\/?[^>]+(>|$)/g, \"\");\r\n\r\n\r\n\r\n// Replace multiple whitespaces with a single space and trim\r\ncleanEmail = cleanEmail.replace(/\\s+/g, ' ').trim();\r\n\r\nreturn cleanEmail"
                },
                "runAfter": {}
            },
            "OpenAI_email_classification": {
                "type": "ServiceProvider",
                "inputs": {
                    "parameters": {
                        "deploymentId": "ssagpt4o",
                        "messages": [
                            {
                                "role": "system",
                                "content": "You are an expert AI assistant specializing in classifying issues face by user in the email.\n\n**Your Task:** Classify email into one of the following categories, sub-categories and provide output only in the json format based on given schema.\n\n- **Categories:**\n  - **PDMLink**: Issues related to PDMLink, OnePDM, or Windchill. Examples include: inability to access these tools or errors related to them.\n  - **License**: Issues related to licensing, such as an inability to start a session, fetch a license, or other license requirements.\n  - **Assist**: Issues requiring assistance in completing a task, such as difficulty installing, accessing certain functionalities, or understanding usage.\n  - **Creo**: Issues related to the Creo tool, such as requests for help with 3D data, surfacing, reverse engineering, or import/export functions.\n  - **None**: In case of no issue\n\n- **Sub-Categories:**\n  - **access**: Issues related to access\n  - **onePDM**: Issues related to error in onePDM, PDMLink or Windchill.\n  - **session**: Issues related to unable to start session\n  - **fetch_license**:Issues related to unable fetch license\n  - **install**: Issues related to unable to install\n  - **functionality**: Issues related to unable to obtain certain functionality\n  - **3D**: When help on 3D data is required\n  - **help_in**: when help in surfing, reverse engineering, import-export etc., required or requested.\n  - **None**: In case of no issue\n\n- **Output Schema:**\n```json\n{  \n  \"type\": \"object\",  \n  \"properties\": {  \n    \"Categories\": {  \n      \"type\": \"string\",  \n      \"enum\": [\"PDMLink\", \"License\", \"Assist\", \"Creo\", \"None\"]  \n    },  \n    \"Sub-Categories\": {  \n      \"type\": \"string\",  \n      \"enum\": [\"access\", \"onePDM\", \"session\", \"fetch_license\", \"install\", \"functionality\", \"3D\", \"help_in\", \"None\"]  \n    }  \n  },  \n  \"required\": [\"Categories\", \"Sub-Categories\"],  \n  \"additionalProperties\": false  \n}\n```"
                            },
                            {
                                "role": "user",
                                "content": "### **Email Subject:**\n@{triggerBody()?['subject']}\n\n###**Email:**\n@{outputs('Execute_JavaScript_Code')}"
                            }
                        ],
                        "temperature": 0.2,
                        "max_tokens": 4000
                    },
                    "serviceProviderConfiguration": {
                        "connectionName": "openai",
                        "operationId": "getChatCompletions",
                        "serviceProviderId": "/serviceProviders/openai"
                    }
                },
                "runAfter": {
                    "Execute_JavaScript_Code": [
                        "SUCCEEDED"
                    ]
                }
            },
            "Execute_JavaScript_Code_1": {
                "type": "JavaScriptCode",
                "inputs": {
                    "code": "var regex = /```json\\s*([\\s\\S]*?)\\s*```/; \r\n\r\nvar regex_new = /```(?:json\\s*)?([\\s\\S]*?)\\s*```/; \r\n\r\nvar output = workflowContext.actions.OpenAI_email_classification.outputs.body.content\r\n\r\nvar match = output.match(regex_new);  \r\nvar jsonString = match ? match[1] : null;\r\n\r\nvar jsonObject = JSON.parse(jsonString);\r\n\r\njsonObject['conversationId'] = workflowContext.trigger.outputs.body.conversationId\r\n\r\njsonObject['subject'] = workflowContext.trigger.outputs.body.subject\r\n\r\njsonObject['to'] = workflowContext.trigger.outputs.body.toRecipients\r\n\r\njsonObject['from'] = workflowContext.trigger.outputs.body.from\r\n\r\njsonObject['date'] = workflowContext.trigger.outputs.body.receivedDateTime\r\n\r\n\r\n// var jsonString = JSON.stringify(jsonObject)\r\n\r\n\r\nvar current_value = workflowContext.actions.Read_blob_content_based_on_URI.outputs.body.content\r\n\r\nif (!Array.isArray(current_value)) {\r\n    current_value = JSON.parse(current_value);\r\n}\r\n\r\ncurrent_value.push(jsonObject);\r\n\r\n// var result = current_value + \"\\n\" + jsonString;\r\n\r\nvar current_jsonString = JSON.stringify(current_value, null, 2);\r\n\r\n\r\nreturn current_jsonString"
                },
                "runAfter": {
                    "Read_blob_content_based_on_URI": [
                        "SUCCEEDED"
                    ]
                }
            },
            "Upload_blob_to_storage_container_based_on_a_URI_1": {
                "type": "ServiceProvider",
                "inputs": {
                    "parameters": {
                        "blobUri": "emails/open_test_clean.json",
                        "content": "@outputs('Execute_JavaScript_Code_1')",
                        "overrideIfExists": "true"
                    },
                    "serviceProviderConfiguration": {
                        "connectionName": "AzureBlob",
                        "operationId": "uploadBlobFromUri",
                        "serviceProviderId": "/serviceProviders/AzureBlob"
                    }
                },
                "runAfter": {
                    "Execute_JavaScript_Code_1": [
                        "SUCCEEDED"
                    ]
                }
            },
            "Read_blob_content_based_on_URI": {
                "type": "ServiceProvider",
                "inputs": {
                    "parameters": {
                        "blobUri": "emails/open_test_clean.json",
                        "inferContentType": true
                    },
                    "serviceProviderConfiguration": {
                        "connectionName": "AzureBlob",
                        "operationId": "readBlobFromUri",
                        "serviceProviderId": "/serviceProviders/AzureBlob"
                    }
                },
                "runAfter": {
                    "OpenAI_email_classification": [
                        "SUCCEEDED"
                    ]
                }
            },
            "Get_an_embedding": {
                "type": "ServiceProvider",
                "inputs": {
                    "parameters": {
                        "deploymentId": "ssaembedding",
                        "input": "@outputs('Execute_JavaScript_Code')"
                    },
                    "serviceProviderConfiguration": {
                        "connectionName": "openai",
                        "operationId": "getSingleEmbedding",
                        "serviceProviderId": "/serviceProviders/openai"
                    }
                },
                "runAfter": {
                    "Execute_JavaScript_Code": [
                        "SUCCEEDED"
                    ]
                }
            },
            "Search_vectors_video": {
                "type": "ServiceProvider",
                "inputs": {
                    "parameters": {
                        "indexName": "vector-1725829103601",
                        "searchVector": {
                            "fieldName": "text_vector",
                            "vector": "@body('Get_an_embedding')['embedding']"
                        },
                        "kNearestNeighbors": 3,
                        "search": "@outputs('Execute_JavaScript_Code')",
                        "filter": "type eq 'video'",
                        "vectorFilterMode": "preFilter",
                        "vectorQueries": [
                            {
                                "kind": "text",
                                "text": "@outputs('Execute_JavaScript_Code')",
                                "fields": "text_vector"
                            }
                        ],
                        "semanticConfiguration": "semantic"
                    },
                    "serviceProviderConfiguration": {
                        "connectionName": "azureaisearch-1_email_2classification_22",
                        "operationId": "vectorSearch",
                        "serviceProviderId": "/serviceProviders/azureaisearch"
                    }
                },
                "runAfter": {
                    "Get_an_embedding": [
                        "SUCCEEDED"
                    ]
                }
            },
            "Upload_blob_to_storage_container_based_on_a_URI": {
                "type": "ServiceProvider",
                "inputs": {
                    "parameters": {
                        "blobUri": "emails/open_rag_clean.txt",
                        "content": "@outputs('formatting_email')",
                        "overrideIfExists": "true"
                    },
                    "serviceProviderConfiguration": {
                        "connectionName": "AzureBlob",
                        "operationId": "uploadBlobFromUri",
                        "serviceProviderId": "/serviceProviders/AzureBlob"
                    }
                },
                "runAfter": {
                    "Reply_to_email_(V3)": [
                        "SUCCEEDED"
                    ]
                }
            },
            "Search_vectors_wiki": {
                "type": "ServiceProvider",
                "inputs": {
                    "parameters": {
                        "indexName": "vector-1725829103601",
                        "searchVector": {
                            "fieldName": "text_vector",
                            "vector": "@body('Get_an_embedding')['embedding']"
                        },
                        "kNearestNeighbors": 3,
                        "search": "@outputs('Execute_JavaScript_Code')",
                        "filter": "type eq 'wiki'",
                        "vectorFilterMode": "preFilter",
                        "vectorQueries": [
                            {
                                "kind": "text",
                                "text": "@outputs('Execute_JavaScript_Code')",
                                "fields": "text_vector"
                            }
                        ],
                        "semanticConfiguration": "semantic"
                    },
                    "serviceProviderConfiguration": {
                        "connectionName": "azureaisearch-1_email_2classification_22",
                        "operationId": "vectorSearch",
                        "serviceProviderId": "/serviceProviders/azureaisearch"
                    }
                },
                "runAfter": {
                    "Get_an_embedding": [
                        "SUCCEEDED"
                    ]
                }
            },
            "Search_vectors_error": {
                "type": "ServiceProvider",
                "inputs": {
                    "parameters": {
                        "indexName": "vector-1725829103601",
                        "searchVector": {
                            "fieldName": "text_vector",
                            "vector": "@body('Get_an_embedding')['embedding']"
                        },
                        "kNearestNeighbors": 2,
                        "search": "@outputs('Execute_JavaScript_Code')",
                        "filter": "type eq 'error'",
                        "vectorFilterMode": "preFilter",
                        "vectorQueries": [
                            {
                                "kind": "text",
                                "text": "@outputs('Execute_JavaScript_Code')",
                                "fields": "text_vector"
                            }
                        ],
                        "semanticConfiguration": "semantic"
                    },
                    "serviceProviderConfiguration": {
                        "connectionName": "azureaisearch-1_email_2classification_22",
                        "operationId": "vectorSearch",
                        "serviceProviderId": "/serviceProviders/azureaisearch"
                    }
                },
                "runAfter": {
                    "Get_an_embedding": [
                        "SUCCEEDED"
                    ]
                }
            },
            "Search_vectors_email": {
                "type": "ServiceProvider",
                "inputs": {
                    "parameters": {
                        "indexName": "vector-1725829103601",
                        "searchVector": {
                            "fieldName": "text_vector",
                            "vector": "@body('Get_an_embedding')['embedding']"
                        },
                        "kNearestNeighbors": 2,
                        "search": "@outputs('Execute_JavaScript_Code')",
                        "filter": "type eq 'email'",
                        "vectorFilterMode": "preFilter",
                        "vectorQueries": [
                            {
                                "kind": "text",
                                "text": "@outputs('Execute_JavaScript_Code')",
                                "fields": "text_vector"
                            }
                        ],
                        "semanticConfiguration": "semantic"
                    },
                    "serviceProviderConfiguration": {
                        "connectionName": "azureaisearch-1_email_2classification_22",
                        "operationId": "vectorSearch",
                        "serviceProviderId": "/serviceProviders/azureaisearch"
                    }
                },
                "runAfter": {
                    "Get_an_embedding": [
                        "SUCCEEDED"
                    ]
                }
            },
            "Search_vectors_creo_view": {
                "type": "ServiceProvider",
                "inputs": {
                    "parameters": {
                        "indexName": "vector-1725829103601",
                        "searchVector": {
                            "fieldName": "text_vector",
                            "vector": "@body('Get_an_embedding')['embedding']"
                        },
                        "kNearestNeighbors": 2,
                        "search": "@outputs('Execute_JavaScript_Code')",
                        "filter": "type eq 'creo_view'",
                        "vectorFilterMode": "preFilter",
                        "vectorQueries": [
                            {
                                "kind": "text",
                                "text": "@outputs('Execute_JavaScript_Code')",
                                "fields": "text_vector"
                            }
                        ],
                        "semanticConfiguration": "semantic"
                    },
                    "serviceProviderConfiguration": {
                        "connectionName": "azureaisearch-1_email_2classification_22",
                        "operationId": "vectorSearch",
                        "serviceProviderId": "/serviceProviders/azureaisearch"
                    }
                },
                "runAfter": {
                    "Get_an_embedding": [
                        "SUCCEEDED"
                    ]
                }
            },
            "Search_vectors_creo_parametric": {
                "type": "ServiceProvider",
                "inputs": {
                    "parameters": {
                        "indexName": "vector-1725829103601",
                        "searchVector": {
                            "fieldName": "text_vector",
                            "vector": "@body('Get_an_embedding')['embedding']"
                        },
                        "kNearestNeighbors": 2,
                        "search": "@outputs('Execute_JavaScript_Code')",
                        "filter": "type eq 'creo_parametric'",
                        "vectorFilterMode": "preFilter",
                        "vectorQueries": [
                            {
                                "kind": "text",
                                "text": "@outputs('Execute_JavaScript_Code')",
                                "fields": "text_vector"
                            }
                        ],
                        "semanticConfiguration": "semantic"
                    },
                    "serviceProviderConfiguration": {
                        "connectionName": "azureaisearch-1_email_2classification_22",
                        "operationId": "vectorSearch",
                        "serviceProviderId": "/serviceProviders/azureaisearch"
                    }
                },
                "runAfter": {
                    "Get_an_embedding": [
                        "SUCCEEDED"
                    ]
                }
            },
            "OpenAI_video_0": {
                "type": "ServiceProvider",
                "inputs": {
                    "parameters": {
                        "deploymentId": "ssagpt4o",
                        "messages": [
                            {
                                "role": "system",
                                "content": "Context information is below.\n---------------------\n@{body('Search_vectors_video')[0]['chunk']}\n---------------------\n\nNote: Context is a transcript, with timestamps in the format **Hours:Minutes:Seconds** or **Minutes:Seconds** on each line above the text. \n\nYou are an expert AI assistant specializing in answering the query with citation and classifying it based on above context and provide output only in the json format based on given Output schema.\n\n**Your Task:**\n1. Given the above context information and not prior knowledge answer the query given by user in step by step format.\n2. Keep your answer concise and solely on the information given in the Context.\n3. classify if the user's query, can be answered using the given Context.\n - \"YES\": if the given Context contains sufficient information to answer the query.\n - \"NO\": if the given Context does not contains sufficient information to answer the query. Class is also \"NO\", if query is too generic or chit-chat e.g, what do you do?, Why am I here, How are you etc.\n4. Always provide all relevant citations at end of the answer, ensuring that the citations are in fomat of List where each elemnt is a timestamp in HH:MM:SS format. e.g, [\"00:11:05\", \"70:02:05\"] or [\"01:14:12\"]. \n5. **Output Schema:**\n```json\n{\n  \"type\": \"object\",  \n  \"properties\": {  \n    \"answer\": {  \n      \"type\": \"string\",  \n      \"description\": \"Answer to the user's query\"  \n    },  \n    \"class_labels\": {  \n      \"type\": \"string\",  \n      \"enum\": [\"YES\", \"NO\"],  \n      \"description\": \"classify if the query, can be answered using the given Context\"  \n    },  \n    \"citations\": {  \n      \"type\": \"array\",  \n      \"items\": {  \n        \"type\": \"string\",  \n        \"format\": \"time\"  // Note: JSON Schema does not have a specific format for timestamps  \n      },  \n      \"default\": [],  \n      \"description\": \"Citations from the answer in form on timestamps HH:MM:SS\"  \n    }  \n  },  \n  \"required\": [\"answer\", \"class_labels\"]  \n}\n```\n\nOutput : \n```json\n{  \n  \"answer\": \"...\",  \n  \"class_labels\": \"...\",  \n  \"citations\": [...]  \n}\n```"
                            },
                            {
                                "role": "user",
                                "content": "@{outputs('Execute_JavaScript_Code')}"
                            }
                        ],
                        "temperature": 0.05,
                        "max_tokens": 4000
                    },
                    "serviceProviderConfiguration": {
                        "connectionName": "openai",
                        "operationId": "getChatCompletions",
                        "serviceProviderId": "/serviceProviders/openai"
                    }
                },
                "runAfter": {
                    "Search_vectors_video": [
                        "SUCCEEDED"
                    ]
                }
            },
            "OpenAI_video_1": {
                "type": "ServiceProvider",
                "inputs": {
                    "parameters": {
                        "deploymentId": "ssagpt4o",
                        "messages": [
                            {
                                "role": "system",
                                "content": "Context information is below.\n---------------------\n@{body('Search_vectors_video')[1]['chunk']}\n---------------------\n\nNote: Context is a transcript, with timestamps in the format **Hours:Minutes:Seconds** or **Minutes:Seconds** on each line above the text. \n\nYou are an expert AI assistant specializing in answering the query with citation and classifying it based on above context and provide output only in the json format based on given Output schema.\n\n**Your Task:**\n1. Given the above context information and not prior knowledge answer the query given by user in step by step format.\n2. Keep your answer concise and solely on the information given in the Context.\n3. classify if the user's query, can be answered using the given Context.\n - \"YES\": if the given Context contains sufficient information to answer the query.\n - \"NO\": if the given Context does not contains sufficient information to answer the query. Class is also \"NO\", if query is too generic or chit-chat e.g, what do you do?, Why am I here, How are you etc.\n4. Always provide all relevant citations at end of the answer, ensuring that the citations are in fomat of List where each elemnt is a timestamp in HH:MM:SS format. e.g, [\"00:11:05\", \"70:02:05\"] or [\"01:14:12\"]. \n5. **Output Schema:**\n```json\n{\n  \"type\": \"object\",  \n  \"properties\": {  \n    \"answer\": {  \n      \"type\": \"string\",  \n      \"description\": \"Answer to the user's query\"  \n    },  \n    \"class_labels\": {  \n      \"type\": \"string\",  \n      \"enum\": [\"YES\", \"NO\"],  \n      \"description\": \"classify if the query, can be answered using the given Context\"  \n    },  \n    \"citations\": {  \n      \"type\": \"array\",  \n      \"items\": {  \n        \"type\": \"string\",  \n        \"format\": \"time\"  // Note: JSON Schema does not have a specific format for timestamps  \n      },  \n      \"default\": [],  \n      \"description\": \"Citations from the answer in form on timestamps HH:MM:SS\"  \n    }  \n  },  \n  \"required\": [\"answer\", \"class_labels\"]  \n}\n```\n\nOutput : \n```json\n{  \n  \"answer\": \"...\",  \n  \"class_labels\": \"...\",  \n  \"citations\": [...]  \n}\n```"
                            },
                            {
                                "role": "user",
                                "content": "@{outputs('Execute_JavaScript_Code')}"
                            }
                        ],
                        "temperature": 0.05,
                        "max_tokens": 4000
                    },
                    "serviceProviderConfiguration": {
                        "connectionName": "openai",
                        "operationId": "getChatCompletions",
                        "serviceProviderId": "/serviceProviders/openai"
                    }
                },
                "runAfter": {
                    "Search_vectors_video": [
                        "SUCCEEDED"
                    ]
                }
            },
            "OpenAI_video_2": {
                "type": "ServiceProvider",
                "inputs": {
                    "parameters": {
                        "deploymentId": "ssagpt4o",
                        "messages": [
                            {
                                "role": "system",
                                "content": "Context information is below.\n---------------------\n@{body('Search_vectors_video')[2]['chunk']}\n---------------------\n\nNote: Context is a transcript, with timestamps in the format **Hours:Minutes:Seconds** or **Minutes:Seconds** on each line above the text. \n\nYou are an expert AI assistant specializing in answering the query with citation and classifying it based on above context and provide output only in the json format based on given Output schema.\n\n**Your Task:**\n1. Given the above context information and not prior knowledge answer the query given by user in step by step format.\n2. Keep your answer concise and solely on the information given in the Context.\n3. classify if the user's query, can be answered using the given Context.\n - \"YES\": if the given Context contains sufficient information to answer the query.\n - \"NO\": if the given Context does not contains sufficient information to answer the query. Class is also \"NO\", if query is too generic or chit-chat e.g, what do you do?, Why am I here, How are you etc.\n4. Always provide all relevant citations at end of the answer, ensuring that the citations are in fomat of List where each elemnt is a timestamp in HH:MM:SS format. e.g, [\"00:11:05\", \"70:02:05\"] or [\"01:14:12\"]. \n5. **Output Schema:**\n```json\n{\n  \"type\": \"object\",  \n  \"properties\": {  \n    \"answer\": {  \n      \"type\": \"string\",  \n      \"description\": \"Answer to the user's query\"  \n    },  \n    \"class_labels\": {  \n      \"type\": \"string\",  \n      \"enum\": [\"YES\", \"NO\"],  \n      \"description\": \"classify if the query, can be answered using the given Context\"  \n    },  \n    \"citations\": {  \n      \"type\": \"array\",  \n      \"items\": {  \n        \"type\": \"string\",  \n        \"format\": \"time\"  // Note: JSON Schema does not have a specific format for timestamps  \n      },  \n      \"default\": [],  \n      \"description\": \"Citations from the answer in form on timestamps HH:MM:SS\"  \n    }  \n  },  \n  \"required\": [\"answer\", \"class_labels\"]  \n}\n```\n\nOutput : \n```json\n{  \n  \"answer\": \"...\",  \n  \"class_labels\": \"...\",  \n  \"citations\": [...]  \n}\n```"
                            },
                            {
                                "role": "user",
                                "content": "@{outputs('Execute_JavaScript_Code')}"
                            }
                        ],
                        "temperature": 0.05,
                        "max_tokens": 4000
                    },
                    "serviceProviderConfiguration": {
                        "connectionName": "openai",
                        "operationId": "getChatCompletions",
                        "serviceProviderId": "/serviceProviders/openai"
                    }
                },
                "runAfter": {
                    "Search_vectors_video": [
                        "SUCCEEDED"
                    ]
                }
            },
            "OpenAI_wiki_0": {
                "type": "ServiceProvider",
                "inputs": {
                    "parameters": {
                        "deploymentId": "ssagpt4o",
                        "messages": [
                            {
                                "role": "system",
                                "content": "Context information is below.\n---------------------\n@{body('Search_vectors_wiki')[0]['chunk']}\n---------------------\n\nYou are an expert AI assistant specializing in answering the query and classifying it based on above contex and provide output only in the json format based on given Output schema.\n\n**Your Task:**\n1. Given the above context information and not prior knowledge answer the query in step by step format.\n2. Keep your answer concise and solely on the information given in the Context.\n3. classify if the query, can be answered using the given Context.\n - \"YES\": if the given Context contains sufficient information to answer the query\n - \"NO\": if the given Context does not contains sufficient information to answer the query. Class is also \"NO\", if query is too generic or chit-chat e.g, what do you do?, Why am I here, How are you etc.\n4. **Output Schema:**\n```json\n{  \n  \"type\": \"object\",  \n  \"properties\": {  \n    \"answer\": {  \n      \"type\": \"string\",  \n      \"description\": \"Answer to the user's query\"  \n    },  \n    \"class_labels\": {  \n      \"type\": \"string\",  \n      \"enum\": [\"YES\", \"NO\"],  \n      \"description\": \"classify if the query, can be answered using the given Context\"  \n    }  \n  },  \n  \"required\": [\"answer\", \"class_labels\"]  \n}  \n```\n\nOutput : \n```json\n{  \n  \"answer\": \"...\",  \n  \"class_labels\": \"...\"\n}\n```"
                            },
                            {
                                "role": "user",
                                "content": "@{outputs('Execute_JavaScript_Code')}"
                            }
                        ],
                        "temperature": 0.05,
                        "max_tokens": 4000
                    },
                    "serviceProviderConfiguration": {
                        "connectionName": "openai",
                        "operationId": "getChatCompletions",
                        "serviceProviderId": "/serviceProviders/openai"
                    }
                },
                "runAfter": {
                    "Search_vectors_wiki": [
                        "SUCCEEDED"
                    ]
                }
            },
            "OpenAI_wiki_1": {
                "type": "ServiceProvider",
                "inputs": {
                    "parameters": {
                        "deploymentId": "ssagpt4o",
                        "messages": [
                            {
                                "role": "system",
                                "content": "Context information is below.\n---------------------\n@{body('Search_vectors_wiki')[1]['chunk']}\n---------------------\n\nYou are an expert AI assistant specializing in answering the query and classifying it based on above contex and provide output only in the json format based on given Output schema.\n\n**Your Task:**\n1. Given the above context information and not prior knowledge answer the query in step by step format.\n2. Keep your answer concise and solely on the information given in the Context.\n3. classify if the query, can be answered using the given Context.\n - \"YES\": if the given Context contains sufficient information to answer the query\n - \"NO\": if the given Context does not contains sufficient information to answer the query. Class is also \"NO\", if query is too generic or chit-chat e.g, what do you do?, Why am I here, How are you etc.\n4. **Output Schema:**\n```json\n{  \n  \"type\": \"object\",  \n  \"properties\": {  \n    \"answer\": {  \n      \"type\": \"string\",  \n      \"description\": \"Answer to the user's query\"  \n    },  \n    \"class_labels\": {  \n      \"type\": \"string\",  \n      \"enum\": [\"YES\", \"NO\"],  \n      \"description\": \"classify if the query, can be answered using the given Context\"  \n    }  \n  },  \n  \"required\": [\"answer\", \"class_labels\"]  \n}  \n```\n\nOutput : \n```json\n{  \n  \"answer\": \"...\",  \n  \"class_labels\": \"...\"\n}\n```"
                            },
                            {
                                "role": "user",
                                "content": "@{outputs('Execute_JavaScript_Code')}"
                            }
                        ],
                        "temperature": 0.05,
                        "max_tokens": 4000
                    },
                    "serviceProviderConfiguration": {
                        "connectionName": "openai",
                        "operationId": "getChatCompletions",
                        "serviceProviderId": "/serviceProviders/openai"
                    }
                },
                "runAfter": {
                    "Search_vectors_wiki": [
                        "SUCCEEDED"
                    ]
                }
            },
            "OpenAI_wiki_2": {
                "type": "ServiceProvider",
                "inputs": {
                    "parameters": {
                        "deploymentId": "ssagpt4o",
                        "messages": [
                            {
                                "role": "system",
                                "content": "Context information is below.\n---------------------\n@{body('Search_vectors_wiki')[2]['chunk']}\n---------------------\n\nYou are an expert AI assistant specializing in answering the query and classifying it based on above contex and provide output only in the json format based on given Output schema.\n\n**Your Task:**\n1. Given the above context information and not prior knowledge answer the query in step by step format.\n2. Keep your answer concise and solely on the information given in the Context.\n3. classify if the query, can be answered using the given Context.\n - \"YES\": if the given Context contains sufficient information to answer the query\n - \"NO\": if the given Context does not contains sufficient information to answer the query. Class is also \"NO\", if query is too generic or chit-chat e.g, what do you do?, Why am I here, How are you etc.\n4. **Output Schema:**\n```json\n{  \n  \"type\": \"object\",  \n  \"properties\": {  \n    \"answer\": {  \n      \"type\": \"string\",  \n      \"description\": \"Answer to the user's query\"  \n    },  \n    \"class_labels\": {  \n      \"type\": \"string\",  \n      \"enum\": [\"YES\", \"NO\"],  \n      \"description\": \"classify if the query, can be answered using the given Context\"  \n    }  \n  },  \n  \"required\": [\"answer\", \"class_labels\"]  \n}  \n```\n\nOutput : \n```json\n{  \n  \"answer\": \"...\",  \n  \"class_labels\": \"...\"\n}\n```"
                            },
                            {
                                "role": "user",
                                "content": "@{outputs('Execute_JavaScript_Code')}"
                            }
                        ],
                        "temperature": 0.05,
                        "max_tokens": 4000
                    },
                    "serviceProviderConfiguration": {
                        "connectionName": "openai",
                        "operationId": "getChatCompletions",
                        "serviceProviderId": "/serviceProviders/openai"
                    }
                },
                "runAfter": {
                    "Search_vectors_wiki": [
                        "SUCCEEDED"
                    ]
                }
            },
            "OpenAI_creo_view_0": {
                "type": "ServiceProvider",
                "inputs": {
                    "parameters": {
                        "deploymentId": "ssagpt4o",
                        "messages": [
                            {
                                "role": "system",
                                "content": "Context information is below.\n---------------------\n@{body('Search_vectors_creo_view')[0]['chunk']}\n---------------------\n\nYou are an expert AI assistant specializing in answering the query and classifying it based on above contex and provide output only in the json format based on given Output schema.\n\n**Your Task:**\n1. Given the above context information and not prior knowledge answer the query in step by step format.\n2. Keep your answer concise and solely on the information given in the Context.\n3. classify if the query, can be answered using the given Context.\n - \"YES\": if the given Context contains sufficient information to answer the query\n - \"NO\": if the given Context does not contains sufficient information to answer the query. Class is also \"NO\", if query is too generic or chit-chat e.g, what do you do?, Why am I here, How are you etc.\n4. **Output Schema:**\n```json\n{  \n  \"type\": \"object\",  \n  \"properties\": {  \n    \"answer\": {  \n      \"type\": \"string\",  \n      \"description\": \"Answer to the user's query\"  \n    },  \n    \"class_labels\": {  \n      \"type\": \"string\",  \n      \"enum\": [\"YES\", \"NO\"],  \n      \"description\": \"classify if the query, can be answered using the given Context\"  \n    }  \n  },  \n  \"required\": [\"answer\", \"class_labels\"]  \n}  \n```\n\nOutput : \n```json\n{  \n  \"answer\": \"...\",  \n  \"class_labels\": \"...\"\n}\n```"
                            },
                            {
                                "role": "user",
                                "content": "@{outputs('Execute_JavaScript_Code')}"
                            }
                        ],
                        "temperature": 0.05,
                        "max_tokens": 4000
                    },
                    "serviceProviderConfiguration": {
                        "connectionName": "openai",
                        "operationId": "getChatCompletions",
                        "serviceProviderId": "/serviceProviders/openai"
                    }
                },
                "runAfter": {
                    "Search_vectors_creo_view": [
                        "SUCCEEDED"
                    ]
                }
            },
            "OpenAI_creo_view_1": {
                "type": "ServiceProvider",
                "inputs": {
                    "parameters": {
                        "deploymentId": "ssagpt4o",
                        "messages": [
                            {
                                "role": "system",
                                "content": "Context information is below.\n---------------------\n@{body('Search_vectors_creo_view')[1]['chunk']}\n---------------------\n\nYou are an expert AI assistant specializing in answering the query and classifying it based on above contex and provide output only in the json format based on given Output schema.\n\n**Your Task:**\n1. Given the above context information and not prior knowledge answer the query in step by step format.\n2. Keep your answer concise and solely on the information given in the Context.\n3. classify if the query, can be answered using the given Context.\n - \"YES\": if the given Context contains sufficient information to answer the query\n - \"NO\": if the given Context does not contains sufficient information to answer the query. Class is also \"NO\", if query is too generic or chit-chat e.g, what do you do?, Why am I here, How are you etc.\n4. **Output Schema:**\n```json\n{  \n  \"type\": \"object\",  \n  \"properties\": {  \n    \"answer\": {  \n      \"type\": \"string\",  \n      \"description\": \"Answer to the user's query\"  \n    },  \n    \"class_labels\": {  \n      \"type\": \"string\",  \n      \"enum\": [\"YES\", \"NO\"],  \n      \"description\": \"classify if the query, can be answered using the given Context\"  \n    }  \n  },  \n  \"required\": [\"answer\", \"class_labels\"]  \n}  \n```\n\nOutput : \n```json\n{  \n  \"answer\": \"...\",  \n  \"class_labels\": \"...\"\n}\n```"
                            },
                            {
                                "role": "user",
                                "content": "@{outputs('Execute_JavaScript_Code')}"
                            }
                        ],
                        "temperature": 0.05,
                        "max_tokens": 4000
                    },
                    "serviceProviderConfiguration": {
                        "connectionName": "openai",
                        "operationId": "getChatCompletions",
                        "serviceProviderId": "/serviceProviders/openai"
                    }
                },
                "runAfter": {
                    "Search_vectors_creo_view": [
                        "SUCCEEDED"
                    ]
                }
            },
            "OpenAI_creo_parametric_0": {
                "type": "ServiceProvider",
                "inputs": {
                    "parameters": {
                        "deploymentId": "ssagpt4o",
                        "messages": [
                            {
                                "role": "system",
                                "content": "Context information is below.\n---------------------\n@{body('Search_vectors_creo_parametric')[0]['chunk']}\n---------------------\n\nYou are an expert AI assistant specializing in answering the query and classifying it based on above contex and provide output only in the json format based on given Output schema.\n\n**Your Task:**\n1. Given the above context information and not prior knowledge answer the query in step by step format.\n2. Keep your answer concise and solely on the information given in the Context.\n3. classify if the query, can be answered using the given Context.\n - \"YES\": if the given Context contains sufficient information to answer the query\n - \"NO\": if the given Context does not contains sufficient information to answer the query. Class is also \"NO\", if query is too generic or chit-chat e.g, what do you do?, Why am I here, How are you etc.\n4. **Output Schema:**\n```json\n{  \n  \"type\": \"object\",  \n  \"properties\": {  \n    \"answer\": {  \n      \"type\": \"string\",  \n      \"description\": \"Answer to the user's query\"  \n    },  \n    \"class_labels\": {  \n      \"type\": \"string\",  \n      \"enum\": [\"YES\", \"NO\"],  \n      \"description\": \"classify if the query, can be answered using the given Context\"  \n    }  \n  },  \n  \"required\": [\"answer\", \"class_labels\"]  \n}  \n```\n\nOutput : \n```json\n{  \n  \"answer\": \"...\",  \n  \"class_labels\": \"...\"\n}\n```"
                            },
                            {
                                "role": "user",
                                "content": "@{outputs('Execute_JavaScript_Code')}"
                            }
                        ],
                        "temperature": 0.05,
                        "max_tokens": 4000
                    },
                    "serviceProviderConfiguration": {
                        "connectionName": "openai",
                        "operationId": "getChatCompletions",
                        "serviceProviderId": "/serviceProviders/openai"
                    }
                },
                "runAfter": {
                    "Search_vectors_creo_parametric": [
                        "SUCCEEDED"
                    ]
                }
            },
            "OpenAI_creo_parametric_1": {
                "type": "ServiceProvider",
                "inputs": {
                    "parameters": {
                        "deploymentId": "ssagpt4o",
                        "messages": [
                            {
                                "role": "system",
                                "content": "Context information is below.\n---------------------\n@{body('Search_vectors_creo_parametric')[1]['chunk']}\n---------------------\n\nYou are an expert AI assistant specializing in answering the query and classifying it based on above contex and provide output only in the json format based on given Output schema.\n\n**Your Task:**\n1. Given the above context information and not prior knowledge answer the query in step by step format.\n2. Keep your answer concise and solely on the information given in the Context.\n3. classify if the query, can be answered using the given Context.\n - \"YES\": if the given Context contains sufficient information to answer the query\n - \"NO\": if the given Context does not contains sufficient information to answer the query. Class is also \"NO\", if query is too generic or chit-chat e.g, what do you do?, Why am I here, How are you etc.\n4. **Output Schema:**\n```json\n{  \n  \"type\": \"object\",  \n  \"properties\": {  \n    \"answer\": {  \n      \"type\": \"string\",  \n      \"description\": \"Answer to the user's query\"  \n    },  \n    \"class_labels\": {  \n      \"type\": \"string\",  \n      \"enum\": [\"YES\", \"NO\"],  \n      \"description\": \"classify if the query, can be answered using the given Context\"  \n    }  \n  },  \n  \"required\": [\"answer\", \"class_labels\"]  \n}  \n```\n\nOutput : \n```json\n{  \n  \"answer\": \"...\",  \n  \"class_labels\": \"...\"\n}\n```"
                            },
                            {
                                "role": "user",
                                "content": "@{outputs('Execute_JavaScript_Code')}"
                            }
                        ],
                        "temperature": 0.05,
                        "max_tokens": 4000
                    },
                    "serviceProviderConfiguration": {
                        "connectionName": "openai",
                        "operationId": "getChatCompletions",
                        "serviceProviderId": "/serviceProviders/openai"
                    }
                },
                "runAfter": {
                    "Search_vectors_creo_parametric": [
                        "SUCCEEDED"
                    ]
                }
            },
            "OpenAI_error_0": {
                "type": "ServiceProvider",
                "inputs": {
                    "parameters": {
                        "deploymentId": "ssagpt4o",
                        "messages": [
                            {
                                "role": "system",
                                "content": "Context information is below.\n---------------------\n@{body('Search_vectors_error')[0]['chunk']}\n---------------------\n\nYou are an expert AI assistant specializing in answering the query and classifying it based on above contex and provide output only in the json format based on given Output schema.\n\n**Your Task:**\n1. Given the above context information and not prior knowledge answer the query in step by step format.\n2. Keep your answer concise and solely on the information given in the Context.\n3. classify if the query, can be answered using the given Context.\n - \"YES\": if the given Context contains sufficient information to answer the query\n - \"NO\": if the given Context does not contains sufficient information to answer the query. Class is also \"NO\", if query is too generic or chit-chat e.g, what do you do?, Why am I here, How are you etc.\n4. **Output Schema:**\n```json\n{  \n  \"type\": \"object\",  \n  \"properties\": {  \n    \"answer\": {  \n      \"type\": \"string\",  \n      \"description\": \"Answer to the user's query\"  \n    },  \n    \"class_labels\": {  \n      \"type\": \"string\",  \n      \"enum\": [\"YES\", \"NO\"],  \n      \"description\": \"classify if the query, can be answered using the given Context\"  \n    }  \n  },  \n  \"required\": [\"answer\", \"class_labels\"]  \n}  \n```\n\nOutput : \n```json\n{  \n  \"answer\": \"...\",  \n  \"class_labels\": \"...\"\n}\n```"
                            },
                            {
                                "role": "user",
                                "content": "@{outputs('Execute_JavaScript_Code')}"
                            }
                        ],
                        "temperature": 0.05,
                        "max_tokens": 4000
                    },
                    "serviceProviderConfiguration": {
                        "connectionName": "openai",
                        "operationId": "getChatCompletions",
                        "serviceProviderId": "/serviceProviders/openai"
                    }
                },
                "runAfter": {
                    "Search_vectors_error": [
                        "SUCCEEDED"
                    ]
                }
            },
            "OpenAI_error_1": {
                "type": "ServiceProvider",
                "inputs": {
                    "parameters": {
                        "deploymentId": "ssagpt4o",
                        "messages": [
                            {
                                "role": "system",
                                "content": "Context information is below.\n---------------------\n@{body('Search_vectors_error')[1]['chunk']}\n---------------------\n\nYou are an expert AI assistant specializing in answering the query and classifying it based on above contex and provide output only in the json format based on given Output schema.\n\n**Your Task:**\n1. Given the above context information and not prior knowledge answer the query in step by step format.\n2. Keep your answer concise and solely on the information given in the Context.\n3. classify if the query, can be answered using the given Context.\n - \"YES\": if the given Context contains sufficient information to answer the query\n - \"NO\": if the given Context does not contains sufficient information to answer the query. Class is also \"NO\", if query is too generic or chit-chat e.g, what do you do?, Why am I here, How are you etc.\n4. **Output Schema:**\n```json\n{  \n  \"type\": \"object\",  \n  \"properties\": {  \n    \"answer\": {  \n      \"type\": \"string\",  \n      \"description\": \"Answer to the user's query\"  \n    },  \n    \"class_labels\": {  \n      \"type\": \"string\",  \n      \"enum\": [\"YES\", \"NO\"],  \n      \"description\": \"classify if the query, can be answered using the given Context\"  \n    }  \n  },  \n  \"required\": [\"answer\", \"class_labels\"]  \n}  \n```\n\nOutput : \n```json\n{  \n  \"answer\": \"...\",  \n  \"class_labels\": \"...\"\n}\n```"
                            },
                            {
                                "role": "user",
                                "content": "@{outputs('Execute_JavaScript_Code')}"
                            }
                        ],
                        "temperature": 0.05,
                        "max_tokens": 4000
                    },
                    "serviceProviderConfiguration": {
                        "connectionName": "openai",
                        "operationId": "getChatCompletions",
                        "serviceProviderId": "/serviceProviders/openai"
                    }
                },
                "runAfter": {
                    "Search_vectors_error": [
                        "SUCCEEDED"
                    ]
                }
            },
            "OpenAI_email_0": {
                "type": "ServiceProvider",
                "inputs": {
                    "parameters": {
                        "deploymentId": "ssagpt4o",
                        "messages": [
                            {
                                "role": "system",
                                "content": "Context information is below.\n---------------------\n@{body('Search_vectors_email')[0]['chunk']}\n---------------------\n\nNote: Context is a email chain that is a discussion between multiple participants.\n\nYou are an expert AI assistant specializing in answering the query and classifying it based on above contex and provide output only in the json format based on given Output schema.\n\n**Your Task:**\n1. Given the above context information and not prior knowledge answer the query in step by step format. \n2. Keep your answer concise and solely on the information given in the Context.\n3. classify if the query, can be answered using the given Context (email chain).\n - \"YES\": if the given Context contains sufficient information to answer the query\n - \"NO\": if the given Context does not contains sufficient information to answer the query. Class is also \"NO\", if query is too generic or chit-chat e.g, what do you do?, Why am I here, How are you etc.\n4. Do not include any name, PII(email,  license code, sever location, important device keys, bank details, cards etc) from the email chain (conversation) while answering the query\n5. **Output Schema:**\n```json\n{  \n  \"type\": \"object\",  \n  \"properties\": {  \n    \"answer\": {  \n      \"type\": \"string\",  \n      \"description\": \"Answer to the user's query\"  \n    },  \n    \"class_labels\": {  \n      \"type\": \"string\",  \n      \"enum\": [\"YES\", \"NO\"],  \n      \"description\": \"classify if the query, can be answered using the given Context\"  \n    }  \n  },  \n  \"required\": [\"answer\", \"class_labels\"]  \n}  \n```\n\nOutput : \n```json\n{  \n  \"answer\": \"...\",  \n  \"class_labels\": \"...\"\n}\n```"
                            },
                            {
                                "role": "user",
                                "content": "@{outputs('Execute_JavaScript_Code')}"
                            }
                        ],
                        "temperature": 0.05,
                        "max_tokens": 4000
                    },
                    "serviceProviderConfiguration": {
                        "connectionName": "openai",
                        "operationId": "getChatCompletions",
                        "serviceProviderId": "/serviceProviders/openai"
                    }
                },
                "runAfter": {
                    "Search_vectors_email": [
                        "SUCCEEDED"
                    ]
                }
            },
            "OpenAI_email_1": {
                "type": "ServiceProvider",
                "inputs": {
                    "parameters": {
                        "deploymentId": "ssagpt4o",
                        "messages": [
                            {
                                "role": "system",
                                "content": "Context information is below.\n---------------------\n@{body('Search_vectors_email')[1]['chunk']}\n---------------------\n\nNote: Context is a email chain that is a discussion between multiple participants.\n\nYou are an expert AI assistant specializing in answering the query and classifying it based on above contex and provide output only in the json format based on given Output schema.\n\n**Your Task:**\n1. Given the above context information and not prior knowledge answer the query in step by step format. \n2. Keep your answer concise and solely on the information given in the Context.\n3. classify if the query, can be answered using the given Context (email chain).\n - \"YES\": if the given Context contains sufficient information to answer the query\n - \"NO\": if the given Context does not contains sufficient information to answer the query. Class is also \"NO\", if query is too generic or chit-chat e.g, what do you do?, Why am I here, How are you etc.\n4. Do not include any name, PII(email,  license code, sever location, important device keys, bank details, cards etc) from the email chain (conversation) while answering the query\n5. **Output Schema:**\n```json\n{  \n  \"type\": \"object\",  \n  \"properties\": {  \n    \"answer\": {  \n      \"type\": \"string\",  \n      \"description\": \"Answer to the user's query\"  \n    },  \n    \"class_labels\": {  \n      \"type\": \"string\",  \n      \"enum\": [\"YES\", \"NO\"],  \n      \"description\": \"classify if the query, can be answered using the given Context\"  \n    }  \n  },  \n  \"required\": [\"answer\", \"class_labels\"]  \n}  \n```\n\nOutput : \n```json\n{  \n  \"answer\": \"...\",  \n  \"class_labels\": \"...\"\n}\n```"
                            },
                            {
                                "role": "user",
                                "content": "@{outputs('Execute_JavaScript_Code')}"
                            }
                        ],
                        "temperature": 0.05,
                        "max_tokens": 4000
                    },
                    "serviceProviderConfiguration": {
                        "connectionName": "openai",
                        "operationId": "getChatCompletions",
                        "serviceProviderId": "/serviceProviders/openai"
                    }
                },
                "runAfter": {
                    "Search_vectors_email": [
                        "SUCCEEDED"
                    ]
                }
            },
            "final_json_for_combined_output": {
                "type": "JavaScriptCode",
                "inputs": {
                    "code": "var regex_new = /```(?:json\\s*)?([\\s\\S]*?)\\s*```/;   \r\n\r\nvar video_0, video_1, video_3, wiki_0, wiki_1, wiki_2, error_0, error_1, email_0, email_1, creo_view_0, creo_view_1, creo_para_0, creo_para_1;  \r\n//                   Video\r\ntry {  \r\n    video_0 =   workflowContext.actions.OpenAI_video_0.outputs.body.content\r\n} catch (error) {  \r\n    video_0 = null;\r\n}\r\n\r\ntry {  \r\n    video_1 =   workflowContext.actions.OpenAI_video_1.outputs.body.content\r\n} catch (error) {  \r\n    video_1 = null;\r\n}\r\n\r\ntry {  \r\n    video_2 =   workflowContext.actions.OpenAI_video_2.outputs.body.content\r\n} catch (error) {  \r\n    video_2 = null;\r\n}\r\n\r\n//                   Wiki\r\n\r\ntry {  \r\n    wiki_0 =   workflowContext.actions.OpenAI_wiki_0.outputs.body.content\r\n} catch (error) {  \r\n    wiki_0 = null;\r\n}\r\n\r\ntry {  \r\n    wiki_1 =   workflowContext.actions.OpenAI_wiki_1.outputs.body.content\r\n} catch (error) {  \r\n    wiki_1 = null;\r\n}\r\n\r\ntry {  \r\n    wiki_0 =   workflowContext.actions.OpenAI_wiki_2.outputs.body.content\r\n} catch (error) {  \r\n    wiki_0 = null;\r\n}\r\n\r\n//                   Email\r\n\r\ntry {  \r\n    email_0 =  workflowContext.actions.OpenAI_email_0.outputs.body.content\r\n} catch (error) {  \r\n    email_0 = null;\r\n}\r\n\r\ntry {  \r\n    email_1 =  workflowContext.actions.OpenAI_email_1.outputs.body.content\r\n} catch (error) {  \r\n    email_1 = null;\r\n}\r\n\r\n//                   Error\r\ntry {  \r\n    error_0 =  workflowContext.actions.OpenAI_error_0.outputs.body.content\r\n} catch (error) {  \r\n    error_0 = null;\r\n}\r\n\r\ntry {  \r\n    error_1 =  workflowContext.actions.OpenAI_error_1.outputs.body.content\r\n} catch (error) {  \r\n    error_1 = null;\r\n}\r\n\r\n//                   Creo view\r\n\r\ntry {  \r\n    creo_view_0 =  workflowContext.actions.OpenAI_creo_view_0.outputs.body.content\r\n} catch (error) {  \r\n    creo_view_0 = null;\r\n}\r\n\r\ntry {  \r\n    creo_view_1 =  workflowContext.actions.OpenAI_creo_view_1.outputs.body.content\r\n} catch (error) {  \r\n    creo_view_1 = null;\r\n}\r\n\r\n//                   Creo Para\r\n\r\ntry {  \r\n    creo_para_0 =  workflowContext.actions.OpenAI_creo_parametric_0.outputs.body.content\r\n} catch (error) {  \r\n    creo_para_0 = null;\r\n}\r\n\r\ntry {  \r\n    creo_para_1 =  workflowContext.actions.OpenAI_creo_parametric_1.outputs.body.content\r\n} catch (error) {  \r\n    creo_para_1 = null;\r\n}\r\n\r\n\r\n// Example usage for multiple outputs\r\n\r\nfunction parseOutputToJson(output) {  \r\n    try {  \r\n        var regex_new = /```(?:json\\s*)?([\\s\\S]*?)\\s*```/;   \r\n        var match = output.match(regex_new);  \r\n        var jsonString = match ? match[1] : null;  \r\n  \r\n    \r\n        // Attempt to parse the JSON string  \r\n        var jsonObject = JSON.parse(jsonString);  \r\n        return jsonObject;\r\n    } catch (e) {  \r\n        // If parsing fails, return the default JSON object  \r\n        return {  \r\n            \"class_labels\": \"NO\",  \r\n            \"citations\": [],  \r\n            \"answer\": \"\"  \r\n        };  \r\n    }  \r\n}\r\n\r\nvideo_0 = parseOutputToJson(video_0);\r\nvideo_1 = parseOutputToJson(video_1);\r\nvideo_2 = parseOutputToJson(video_2);\r\n\r\nwiki_0 = parseOutputToJson(wiki_0);\r\nwiki_1 = parseOutputToJson(wiki_1);\r\nwiki_2 = parseOutputToJson(wiki_2);\r\n\r\nemail_0 = parseOutputToJson(email_0);\r\nemail_1 = parseOutputToJson(email_1);\r\n\r\nerror_0 = parseOutputToJson(error_0);\r\nerror_1 = parseOutputToJson(error_1);\r\n\r\ncreo_view_0 = parseOutputToJson(creo_view_0);\r\ncreo_view_1 = parseOutputToJson(creo_view_1);\r\n\r\ncreo_para_0 = parseOutputToJson(creo_para_0);\r\ncreo_para_1 = parseOutputToJson(creo_para_1);\r\n\r\n// Function to append a value if it's not already in the array\r\n\r\nvar final_citations = [];\r\nvar combined_answer = \"\";\r\nvar video_ans = \"\";\r\nvar wiki_ans = \"\";\r\nvar error_ans = \"\";\r\nvar email_ans = \"\";\r\nvar creo_view_ans = \"\";\r\nvar creo_para_ans = \"\";\r\n\r\n\r\nfunction appendIfNotPresent(url, name) {  \r\n    // Check if the URL is already in the final_citations  \r\n    var exists = final_citations.some(function(element) {  \r\n        return element[0] === url; // Check if the first element (URL) matches  \r\n    });  \r\n  \r\n    // If the URL is not present, append the new element  \r\n    if (!exists) {  \r\n        final_citations.push([url, name]); // Append as an array [url, name]  \r\n    }  \r\n}\r\n\r\nif (video_0.class_labels === \"YES\") {\r\n    appendIfNotPresent(workflowContext.actions.Search_vectors_video.outputs.body[0]['url_metadata'], workflowContext.actions.Search_vectors_video.outputs.body[0]['file_name_metadata'].split('.')[0])\r\n    video_ans = video_ans + '\\n\\n'+ video_0.answer \r\n}\r\n\r\nif (video_1.class_labels === \"YES\") {\r\n    appendIfNotPresent(workflowContext.actions.Search_vectors_video.outputs.body[1]['url_metadata'], workflowContext.actions.Search_vectors_video.outputs.body[1]['file_name_metadata'].split('.')[0])\r\n    video_ans = video_ans+'\\n\\n' + video_1.answer   \r\n}\r\n\r\nif (video_2.class_labels === \"YES\") {\r\n    appendIfNotPresent(workflowContext.actions.Search_vectors_video.outputs.body[2]['url_metadata'], workflowContext.actions.Search_vectors_video.outputs.body[2]['file_name_metadata'].split('.')[0])\r\n    video_ans = video_ans + '\\n\\n' +video_2.answer      \r\n}\r\n\r\nif (wiki_0.class_labels === \"YES\") {\r\n    appendIfNotPresent(workflowContext.actions.Search_vectors_wiki.outputs.body[0]['url_metadata'], workflowContext.actions.Search_vectors_wiki.outputs.body[0]['file_name_metadata'].split('.')[0])\r\n    wiki_ans = wiki_ans + '\\n\\n'+ wiki_0.answer \r\n}\r\n\r\nif (wiki_1.class_labels === \"YES\") {\r\n    appendIfNotPresent(workflowContext.actions.Search_vectors_wiki.outputs.body[1]['url_metadata'], workflowContext.actions.Search_vectors_wiki.outputs.body[1]['file_name_metadata'].split('.')[0])\r\n    wiki_ans = wiki_ans + '\\n\\n'+ wiki_1.answer \r\n}\r\n\r\nif (wiki_2.class_labels === \"YES\") {\r\n    appendIfNotPresent(workflowContext.actions.Search_vectors_wiki.outputs.body[2]['url_metadata'], workflowContext.actions.Search_vectors_wiki.outputs.body[2]['file_name_metadata'].split('.')[0])\r\n    wiki_ans = wiki_ans + '\\n\\n'+ wiki_2.answer \r\n}\r\n\r\n\r\nif (error_0.class_labels === \"YES\") {\r\n    appendIfNotPresent(workflowContext.actions.Search_vectors_error.outputs.body[0]['url_metadata'], workflowContext.actions.Search_vectors_error.outputs.body[0]['file_name_metadata'].split('.')[0])\r\n    error_ans = error_ans + '\\n\\n'+ error_0.answer \r\n}\r\n\r\nif (error_1.class_labels === \"YES\") {\r\n    appendIfNotPresent(workflowContext.actions.Search_vectors_error.outputs.body[1]['url_metadata'], workflowContext.actions.Search_vectors_error.outputs.body[1]['file_name_metadata'].split('.')[0])\r\n    error_ans = error_ans + '\\n\\n'+ error_1.answer \r\n}\r\n\r\n\r\nif (email_0.class_labels === \"YES\") {\r\n    appendIfNotPresent(workflowContext.actions.Search_vectors_email.outputs.body[0]['url_metadata'], workflowContext.actions.Search_vectors_email.outputs.body[0]['file_name_metadata'].split('.')[0])\r\n    email_ans = email_ans + '\\n\\n'+ email_0.answer \r\n}\r\n\r\nif (email_1.class_labels === \"YES\") {\r\n    appendIfNotPresent(workflowContext.actions.Search_vectors_email.outputs.body[1]['url_metadata'], workflowContext.actions.Search_vectors_email.outputs.body[1]['file_name_metadata'].split('.')[0])\r\n    email_ans = email_ans + '\\n\\n'+ email_1.answer \r\n}\r\n\r\nif (creo_view_0.class_labels === \"YES\") {\r\n    appendIfNotPresent(workflowContext.actions.Search_vectors_creo_view.outputs.body[0]['url_metadata'], workflowContext.actions.Search_vectors_creo_view.outputs.body[0]['file_name_metadata'].split('.')[0])\r\n    creo_view_ans = creo_view_ans + '\\n\\n'+ creo_view_0.answer \r\n}\r\n\r\nif (creo_view_1.class_labels === \"YES\") {\r\n    appendIfNotPresent(workflowContext.actions.Search_vectors_creo_view.outputs.body[1]['url_metadata'], workflowContext.actions.Search_vectors_creo_view.outputs.body[1]['file_name_metadata'].split('.')[0])\r\n    creo_view_ans = creo_view_ans + '\\n\\n'+ creo_view_1.answer \r\n}\r\n\r\n\r\nif (creo_para_0.class_labels === \"YES\") {\r\n    appendIfNotPresent(workflowContext.actions.Search_vectors_creo_parametric.outputs.body[0]['url_metadata'], workflowContext.actions.Search_vectors_creo_parametric.outputs.body[0]['file_name_metadata'].split('.')[0])\r\n    creo_para_ans = creo_para_ans + '\\n\\n'+ creo_para_0.answer \r\n}\r\n\r\nif (creo_para_1.class_labels === \"YES\") {\r\n    appendIfNotPresent(workflowContext.actions.Search_vectors_creo_parametric.outputs.body[1]['url_metadata'], workflowContext.actions.Search_vectors_creo_parametric.outputs.body[1]['file_name_metadata'].split('.')[0])\r\n    creo_para_ans = creo_para_ans + '\\n\\n'+ creo_para_1.answer \r\n}\r\n\r\ncombined_answer = `\r\nAnswer from source: **Video**\r\n${video_ans}\r\n\r\n\r\nAnswer from source: **Wiki**\r\n${wiki_ans}\r\n\r\n\r\nAnswer from source: **Error Documents**\r\n${error_ans}\r\n\r\n\r\nAnswer from source: **Email Chain**\r\n${email_ans}\r\n\r\n\r\nAnswer from source: **Creo Parametric**\r\n${creo_para_ans}\r\n\r\n\r\nAnswer from source: **Creo View**\r\n${creo_view_ans}\r\n`\r\n\r\nvar rag_user_query = `\r\nAnswer's from the different source.\r\n------------------------------------------\r\n${combined_answer}\r\n------------------------------------------\r\n\r\n\r\n**Query:** \r\n${workflowContext.actions.Execute_JavaScript_Code.outputs}\r\n`\r\n\r\n  \r\n// You can return or process the outputs further as needed  \r\nreturn {  \r\n    \"rag_user_query\": rag_user_query,  \r\n    \"final_citations\": final_citations,   \r\n};  "
                },
                "runAfter": {
                    "OpenAI_video_0": [
                        "SUCCEEDED",
                        "TIMEDOUT",
                        "SKIPPED",
                        "FAILED"
                    ],
                    "OpenAI_video_1": [
                        "SUCCEEDED",
                        "TIMEDOUT",
                        "SKIPPED",
                        "FAILED"
                    ],
                    "OpenAI_wiki_0": [
                        "SUCCEEDED",
                        "TIMEDOUT",
                        "SKIPPED",
                        "FAILED"
                    ],
                    "OpenAI_wiki_1": [
                        "SUCCEEDED",
                        "TIMEDOUT",
                        "SKIPPED",
                        "FAILED"
                    ],
                    "OpenAI_wiki_2": [
                        "SUCCEEDED",
                        "TIMEDOUT",
                        "SKIPPED",
                        "FAILED"
                    ],
                    "OpenAI_video_2": [
                        "SUCCEEDED",
                        "TIMEDOUT",
                        "SKIPPED",
                        "FAILED"
                    ],
                    "OpenAI_creo_view_0": [
                        "SUCCEEDED",
                        "TIMEDOUT",
                        "SKIPPED",
                        "FAILED"
                    ],
                    "OpenAI_creo_view_1": [
                        "SUCCEEDED",
                        "TIMEDOUT",
                        "SKIPPED",
                        "FAILED"
                    ],
                    "OpenAI_creo_parametric_0": [
                        "SUCCEEDED",
                        "TIMEDOUT",
                        "SKIPPED",
                        "FAILED"
                    ],
                    "OpenAI_creo_parametric_1": [
                        "SUCCEEDED",
                        "TIMEDOUT",
                        "SKIPPED",
                        "FAILED"
                    ],
                    "OpenAI_error_0": [
                        "SUCCEEDED",
                        "TIMEDOUT",
                        "SKIPPED",
                        "FAILED"
                    ],
                    "OpenAI_error_1": [
                        "SUCCEEDED",
                        "TIMEDOUT",
                        "SKIPPED",
                        "FAILED"
                    ],
                    "OpenAI_email_0": [
                        "SUCCEEDED",
                        "TIMEDOUT",
                        "SKIPPED",
                        "FAILED"
                    ],
                    "OpenAI_email_1": [
                        "SUCCEEDED",
                        "TIMEDOUT",
                        "SKIPPED",
                        "FAILED"
                    ]
                }
            },
            "OpenAI_Combine_answer": {
                "type": "ServiceProvider",
                "inputs": {
                    "parameters": {
                        "deploymentId": "ssagpt4o",
                        "messages": [
                            {
                                "role": "system",
                                "content": "**Task:** Generate a comprehensive answer by synthesizing responses from all available sources.\n\n1. **Process each source individually** following the specified **priority order**:\n   - **Video** (highest priority)\n   - **Wiki**\n   - **Error Documents**\n   - **Email Chain**\n   - **Creo Parametric**\n   - **Creo View** (lowest priority)\n\n2. **Adhere strictly to the priority order** when crafting the final answer.\n\n3. **Fallback Condition**: If none of the sources provide an answer, respond based on general knowledge, clarifying that no information was found in the provided sources.\n\n4. Do not mentioned any sources name in the final answer."
                            },
                            {
                                "role": "user",
                                "content": "@outputs('final_json_for_combined_output')['rag_user_query']"
                            }
                        ],
                        "temperature": 0.2,
                        "max_tokens": 4000
                    },
                    "serviceProviderConfiguration": {
                        "connectionName": "openai",
                        "operationId": "getChatCompletions",
                        "serviceProviderId": "/serviceProviders/openai"
                    }
                },
                "runAfter": {
                    "final_json_for_combined_output": [
                        "SUCCEEDED"
                    ]
                }
            },
            "OpenAI_Formatting_the_response": {
                "type": "ServiceProvider",
                "inputs": {
                    "parameters": {
                        "deploymentId": "ssagpt4o",
                        "messages": [
                            {
                                "role": "system",
                                "content": "@{outputs('formatting_response_json')}"
                            },
                            {
                                "role": "user",
                                "content": "**query:** \n @{outputs('Execute_JavaScript_Code')} \n\n\n **Answer from RAG:**\n @{body('OpenAI_Combine_answer')?['content']}"
                            }
                        ],
                        "temperature": 0.2,
                        "max_tokens": 4000
                    },
                    "serviceProviderConfiguration": {
                        "connectionName": "openai",
                        "operationId": "getChatCompletions",
                        "serviceProviderId": "/serviceProviders/openai"
                    }
                },
                "runAfter": {
                    "formatting_response_json": [
                        "SUCCEEDED"
                    ]
                }
            },
            "formatting_response_json": {
                "type": "JavaScriptCode",
                "inputs": {
                    "code": "var citation_list = workflowContext.actions.final_json_for_combined_output.outputs['final_citations']\r\n\r\nif (citation_list.length > 0) {  \r\n    var system_prompt = `**Instruction for Generating and Formatting the Response**   \r\n1. **Answer from RAG** is a correct answer to unser's query, therefore in response just use re-write **Answer from RAG** without referencing other sources or prior knowledge.  \r\n2. Re-Write **Answer from RAG** in a step-by-step format, enhancing readability and making it more engaging without altering the original content. Structure each step clearly, using bullet points or numbered steps if appropriate.\r\n3. if **Answer from RAG** contains this \"the context provided does not provided the specific detais\", \"There is no answer available\" or similar then do not re-write **Answer from RAG**, Just state There isn't an available answer at the moment, but I've included a few articles in the citations that may be helpful or something similar in step by step format \r\n` \r\n} else {  \r\n    var system_prompt = `For every user query you always give this response 'There is no answer available', except Hi, hello, why am I here, query about you, greeting queries or similar.\r\n\r\nExample 1:\r\nQuery: Hi.\r\nYour response: Hello! How can I assist you today?\r\n\r\nExample 2:\r\nQuery: How to play football?\r\nYour response: There is no answer available\r\n\r\nExample 3:\r\nQuery: What do you do?\r\nYour response: I assist with a wide range of tasks, from **answering questions** and **providing guidance** to helping with **technical issues**, and more.\r\n\r\nExample 4:\r\nQuery: My pc is running slow?\r\nYour response: **There is no answer available**\r\n\r\n\r\nExample 5:\r\nQuery: Why am I here?\r\nYour response: You're here because you might need assistance with **technical** or **IT-related** questions, particularly those related to **design**, **tools**, or **troubleshooting** within your work environment. \\n\\nLet me know if there's something specific I can help you with!\r\n\r\nExample 6:\r\n\r\nYour response: \r\nI can assist you by:\r\n1. **Answering Queries on Document Design and Tools**\r\n2. **Technical Support**\r\n3. **Best Practices and Workflow Guidance**\r\nLet me know how I can specifically help with your current project or challenge!\r\n`\r\n}\r\n\r\nreturn system_prompt"
                },
                "runAfter": {
                    "OpenAI_Combine_answer": [
                        "SUCCEEDED"
                    ]
                }
            },
            "formatting_email": {
                "type": "JavaScriptCode",
                "inputs": {
                    "code": "function run(response, citation) {\r\n  // Retrieve the markdown text and the citation list from the input.\r\n  var text_md = response;\r\n  var citation_list = citation;\r\n\r\n  // A simple Markdown-to-HTML converter.\r\n  // Note: This function handles basic Markdown features.\r\n  function convertMarkdownToHtml(md) {\r\n    var html = md;\r\n    // Convert headers (# to <h1>, ## to <h2>, etc.)\r\n    html = html.replace(/^###### (.*$)/gim, '<h6>$1</h6>');\r\n    html = html.replace(/^##### (.*$)/gim, '<h5>$1</h5>');\r\n    html = html.replace(/^#### (.*$)/gim, '<h4>$1</h4>');\r\n    html = html.replace(/^### (.*$)/gim, '<h3>$1</h3>');\r\n    html = html.replace(/^## (.*$)/gim, '<h2>$1</h2>');\r\n    html = html.replace(/^# (.*$)/gim, '<h1>$1</h1>');\r\n    \r\n    // Convert bold text (**text**)\r\n    html = html.replace(/\\*\\*(.+?)\\*\\*/gim, '<b>$1</b>');\r\n    // Convert italic text (*text*)\r\n    html = html.replace(/\\*(.+?)\\*/gim, '<i>$1</i>');\r\n    // Convert Markdown links [text](url) to HTML links\r\n    html = html.replace(/\\[([^\\]]+)\\]\\(([^)]+)\\)/gim, '<a href=\"$2\">$1</a>');\r\n    // Convert line breaks\r\n    html = html.replace(/\\n/g, '<br>');\r\n    \r\n    return html.trim();\r\n  }\r\n  \r\n  // Convert the Markdown to HTML.\r\n  var htmlBody = convertMarkdownToHtml(text_md);\r\n  \r\n  // If there are citations, add a \"Citations\" section.\r\n  if (citation_list && citation_list.length > 0) {\r\n    htmlBody += '<br><br>'; // add some spacing\r\n    htmlBody += '<h3>Citations</h3>';\r\n    htmlBody += '<ul>';\r\n    \r\n    // Loop over each citation and append as list item.\r\n    for (var i = 0; i < citation_list.length; i++) {\r\n      var citation = citation_list[i];\r\n      var url = citation[0];\r\n      var urlName = citation[1];\r\n      htmlBody += '<li><a href=\"' + url + '\">' + urlName + '</a></li>';\r\n    }\r\n    \r\n    htmlBody += '</ul>';\r\n  }\r\n  htmlBody += '<br><br>';\r\n  htmlBody += 'Regards,<br>';\r\n  htmlBody += 'Smart Service Assistant';\r\n  \r\n  // Return the formatted HTML body.\r\n  return { \"email_body\": htmlBody };\r\n}\r\n\r\nvar email_body = run(workflowContext.actions.OpenAI_Formatting_the_response.outputs.body.content,\r\nworkflowContext.actions.final_json_for_combined_output.outputs['final_citations'])\r\n\r\nreturn email_body['email_body']\r\n\r\n\r\n\r\n"
                },
                "runAfter": {
                    "OpenAI_Formatting_the_response": [
                        "SUCCEEDED"
                    ]
                }
            },
            "Reply_to_email_(V3)": {
                "type": "ApiConnection",
                "inputs": {
                    "host": {
                        "connection": {
                            "referenceName": "office365-1"
                        }
                    },
                    "method": "post",
                    "body": {
                        "To": "@triggerBody()?['from']",
                        "Cc": "@triggerBody()?['ccRecipients']",
                        "Subject": "Re: @{triggerBody()?['subject']}",
                        "Body": "<p class=\"editor-paragraph\">@{outputs('formatting_email')}</p><br>",
                        "ReplyAll": true
                    },
                    "path": "/v3/Mail/ReplyTo/@{encodeURIComponent(triggerBody()?['id'])}"
                },
                "runAfter": {
                    "formatting_email": [
                        "SUCCEEDED"
                    ]
                }
            }
        },
        "contentVersion": "1.0.0.0",
        "outputs": {},
        "triggers": {
            "When_a_new_email_arrives_(V3)": {
                "type": "ApiConnectionNotification",
                "inputs": {
                    "host": {
                        "connection": {
                            "referenceName": "office365-1"
                        }
                    },
                    "fetch": {
                        "pathTemplate": {
                            "template": "/v3/Mail/OnNewEmail"
                        },
                        "method": "get",
                        "queries": {
                            "to": "dets-ssa@microsoft.com",
                            "subjectFilter": "wct",
                            "folderPath": "Inbox"
                        }
                    },
                    "subscribe": {
                        "body": {
                            "NotificationUrl": "@listCallbackUrl()"
                        },
                        "pathTemplate": {
                            "template": "/GraphMailSubscriptionPoke/$subscriptions"
                        },
                        "method": "post",
                        "queries": {
                            "folderPath": "Inbox"
                        }
                    }
                },
                "splitOn": "@triggerBody()?['value']"
            }
        }
    },
    "kind": "stateful"
}