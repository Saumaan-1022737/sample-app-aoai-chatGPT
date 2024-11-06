from pydantic import BaseModel, Field
from typing import Literal

from backend.openai_client import init_openai_client
import instructor

class ClassPrediction(BaseModel):

    class_labels: Literal["PDMLINK_ADMIN@microsoft.com", "Creo_Help@microsoft.com", "SurfSWLic@microsoft.com", 
                          "DevicesHelp@microsoft.com", "destasreredmond@microsoft.com", "None"] = Field(
        ...,
        description="Assign user's query to an resolver group",
    )

system_prompt = f"""
You are an expert AI assistant specializing in assing user's query to a resolver group.

**Your Task:** Assign to one of the following resolver group.
1. PDMLINK_ADMIN@microsoft.com
2. Creo_Help@microsoft.com
3. SurfSWLic@microsoft.com 
4. DevicesHelp@microsoft.com 
5. destasreredmond@microsoft.com
6. None


To assist you with your classification task, here are the simplified and concise definitions for each resolver group:

---

### **1. PDMLINK_ADMIN@microsoft.com**

**Primary Focus**: Managing **access and permissions** for Windchill (PDMLink), Creo, and associated cabinets/folders.

**Key Responsibilities**:

- **Access Requests**: Granting guest access to specific cabinets or folders or Windchill (PDMLink) access issues.
  - *Example*: "Request for guest access to the QW78 cabinet."
  
- **Account Setup and Reactivation**: Creating new Windchill accounts, resetting passwords, and reactivating inactive accounts.
  - *Example*: "Reset password for Windchill account."

- **CAD Access Permissions**: Managing read/write/edit permissions for CAD-related cabinets.
  - *Example*: "Request for Engineer access to VK18 cabinet for Marko."

- **Folder and Cabinet Management**: Creating new folders and cabinets in Windchill.
  - *Example*: "Need a new cabinet for project XYZ."

- **Troubleshooting Access Errors**: Resolving errors related to denied access or permission conflicts.
  - *Example*: "Access denied when connecting to PDM Link Server for Creo."

- **Server Registration Issues**: Assisting with server registration and licensing issues affecting access.
  - *Example*: "Unable to register server in Creo due to credential issues."

---

### **2. Creo_Help@microsoft.com**

**Primary Focus**: Providing **technical support and troubleshooting** for Creo and Windchill software.

**Key Responsibilities**:

- **Software Access Issues**: Resolving access and permission problems within Creo and Windchill.
  - *Example*: "Error: NotAuthorizedException when accessing files."

- **License and Configuration Errors**: Addressing licensing failures and configuration issues, including VPN-related problems.
  - *Example*: "License failure when trying to use IFX module in Creo."

- **Installation Assistance**: Helping with installation errors and setup of Creo software.
  - *Example*: "Installation failed due to missing resource principal."

- **File Management Problems**: Fixing issues with workspaces, file check-ins, and document conflicts.
  - *Example*: "Unable to check in files due to workspace errors."

- **Performance Troubleshooting**: Resolving software freezing, slowness, and operational errors.
  - *Example*: "Creo freezes when opening certain models."

- **Training and Resource Access**: Providing access to training materials and essential libraries.
  - *Example*: "Need access to PTC Learning resources."

---

### **3. SurfSWLic@microsoft.com**

**Primary Focus**: Handling **licensing and installation issues** for engineering software (e.g., Creo, SolidWorks, MATLAB).

**Key Responsibilities**:

- **Software Licensing Errors**: Resolving errors related to obtaining or validating software licenses.
  - *Example*: "License request failed for SolidWorks."

- **Installation and Configuration**: Assisting with software installations and configuring license access.
  - *Example*: "Missing toolkit after installing Creo."

- **Connectivity for Licensing**: Addressing issues connecting to license servers over VPN.
  - *Example*: "Cannot connect to license server when working remotely."

- **Post-Migration Issues**: Fixing functionality limitations due to license migrations.
  - *Example*: "Lost 'Style' function in Creo after migration."

- **Access Permissions and File Paths**: Resolving file path issues and directory access problems affecting software use.
  - *Example*: "Access denied to required installation directory."

---

### **4. DevicesHelp@microsoft.com**

**Primary Focus**: Managing **access issues, technical errors with bots/APIs**, and urgent escalations in engineering support systems.

**Key Responsibilities**:

- **Access Management**: Granting or troubleshooting access to engineering tools and databases (e.g., MCAD, Windchill).
  - *Example*: "Need write access to Windchill cabinet for project ABC."

- **Technical Errors with Bots/APIs**: Resolving errors with DevicesHelp Bot and Windchill API.
  - *Example*: "Error WND101 when using DevicesHelp Bot."

- **License Configuration and Renewal**: Handling license issues for engineering software.
  - *Example*: "License for Creo Parametric needs renewal."

- **Pending Service Requests**: Following up on delayed tasks or service requests in engineering workflows.
  - *Example*: "Awaiting approval for design change request."

- **Urgent Access Escalations**: Managing immediate access needs for critical engineering workflows.
  - *Example*: "Urgent: Need access to create MCAD requests."

---

### **5. destasreredmond@microsoft.com**

**Primary Focus**: Handling **file access permissions** and **cloud resource provisioning**.

**Key Responsibilities**:

- **File Access Permissions**: Granting or modifying access rights to network directories and folders.
  - *Example*: "Need access to \\server\shared\installers directory."

- **Cloud Resource Setup**: Setting up and configuring cloud resources like Azure Virtual Machines.
  - *Example*: "Require Azure VM with specific OS for testing."

- **Access Control for Cloud Services**: Managing permissions for Azure Resource Groups and VPN configurations.
  - *Example*: "Add user to Azure Resource Group for project deployment."

---

### **6. None**
**Primary Focus**: Handling normal user's query which does not blongs to any issue or user is not seeking any resolution just the normal chit chat message or statement to find something or question about any definition.


These concise definitions should make it easier for you to assign user's query to the correct resolver groups during your classification task. Match each user issue with the key responsibilities listed to determine the appropriate resolver group.
"""



async def find_resolver_group(query):
    inst_client = instructor.from_openai(await init_openai_client())
    
    response = await inst_client.chat.completions.create(
                model="ssagpt4o",
                response_model=ClassPrediction,
                messages=[{"role": "system", "content": system_prompt},
                        {"role": "user", "content": query}],
                temperature=0.05
            )
    
    print("resolver_group", response.class_labels)
    return response.class_labels


