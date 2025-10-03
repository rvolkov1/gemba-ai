# System Context Diagram

This diagram shows the high-level context of the Gemba AI system, its users, and the external systems it interacts with.

```mermaid
graph TD
    subgraph "Users"
        cloud_user[Cloud User]
        local_user[Local User]
    end

    subgraph "Gemba AI System"
        gemba_ai_system[Your Application]
    end

    subgraph "External Systems"
        cognito[AWS Cognito]
        s3[AWS S3]
    end

    cloud_user -- "Uses CLI or future web interface" --> gemba_ai_system
    local_user -- "Uses for local development" --> gemba_ai_system

    gemba_ai_system -- "Authenticates cloud users via" --> cognito
    gemba_ai_system -- "Stores videos & results for cloud users in" --> s3

```

## Description

*   **Users:** There are two main types of users. The `Cloud User` who interacts with the production system, and the `Local User` (developer) who runs the system on their own machine.
*   **Gemba AI System:** This is the system you are building, composed of all your microservices.
*   **External Systems:**
    *   **AWS Cognito:** Used for handling user sign-up and login for the cloud deployment.
    *   **AWS S3:** The cloud object storage for videos and processing results.
