# Mermaid Export Pro - Comprehensive Debug Report

**Timestamp**: 2025-10-14T10:07:00.570Z

**Test Coverage**: 10 diagram types × 2 complexity levels × 3 formats × 2 strategies

**Strategies Tested**: 
- CLI Export (using @mermaid-js/mermaid-cli)
- Web Export (VS Code webview + mermaid.js)

## Flowchart Test Diagrams

### Simple Version
```mermaid
flowchart TD
    A[Start] --> B{Decision}
    B -->|Yes| C[Action 1]
    B -->|No| D[Action 2]
    C --> E[End]
    D --> E
```

### Complex Version  
```mermaid
flowchart TB
    A[Christmas] -->|Get money| B(Go shopping)
    B --> C{Let me think}
    C -->|One| D[Laptop]
    C -->|Two| E[iPhone]
    C -->|Three| F[fa:fa-car Car]
    D --> G[Result one]
    E --> G
    F --> G
    G --> H{Another decision}
    H -->|Choice 1| I[Option A]
    H -->|Choice 2| J[Option B]
    H -->|Choice 3| K[Option C]
    I --> L((Final))
    J --> L
    K --> L
    L --> M[End Process]
    
    subgraph "Subprocess"
        N[Sub Start] --> O{Sub Decision}
        O -->|Yes| P[Sub Action]
        O -->|No| Q[Alternative]
        P --> R[Sub End]
        Q --> R
    end
    
    M --> N
```

## Sequence Test Diagrams

### Simple Version
```mermaid
sequenceDiagram
    participant Alice
    participant Bob
    Alice->>Bob: Hello Bob, how are you?
    Bob-->>Alice: Great!
```

### Complex Version  
```mermaid
sequenceDiagram
    participant Browser
    participant API Gateway
    participant Auth Service
    participant User Service
    participant Database
    participant Cache
    
    Browser->>+API Gateway: Login Request
    API Gateway->>+Auth Service: Validate Credentials
    Auth Service->>+Database: Query User
    Database-->>-Auth Service: User Data
    Auth Service->>+Cache: Store Session
    Cache-->>-Auth Service: Session ID
    Auth Service-->>-API Gateway: JWT Token
    API Gateway-->>-Browser: Login Success
    
    Browser->>+API Gateway: Get Profile
    API Gateway->>+Auth Service: Validate JWT
    Auth Service->>+Cache: Check Session
    Cache-->>-Auth Service: Valid Session
    Auth Service-->>-API Gateway: Authorized
    API Gateway->>+User Service: Get User Profile
    User Service->>+Database: Query Profile
    Database-->>-User Service: Profile Data
    User Service-->>-API Gateway: Profile Response
    API Gateway-->>-Browser: User Profile
    
    Note over Browser,Database: Authentication Flow
    Note right of Cache: Session expires in 24h
    
    loop Health Check
        API Gateway->>Database: ping
        Database-->>API Gateway: pong
    end
```

## ClassDiagram Test Diagrams

### Simple Version
```mermaid
classDiagram
    class Animal {
        +String name
        +makeSound()
    }
    class Dog {
        +bark()
    }
    Animal <|-- Dog
```

### Complex Version  
```mermaid
classDiagram
    class Animal {
        +String name
        +int age
        +String species
        +makeSound() String
        +calculateLifeExpectancy() Integer
        +updateHealthRecord(record) Boolean
        +getMedicalHistory() List~MedicalRecord~
    }
    
    class Dog {
        +String breed
        +String ownerName
        +Boolean isTrainingComplete
        +bark() String
        +fetchBall() Boolean
        +performTrick(trickName) String
        +calculateWalkDistance() Double
    }
    
    class Cat {
        +Boolean indoor
        +String favoriteSpot
        +Integer huntingSkillLevel
        +meow() String
        +purr() Boolean
        +huntMice() Integer
        +climbTree() Boolean
    }
    
    class VeterinaryRecord {
        +Date lastCheckup
        +List~String~ vaccinations
        +String veterinarianName
        +Double weight
        +generateReport() String
        +scheduleNextAppointment() Date
        +updateVaccination(vaccine) Boolean
    }
    
    class Owner {
        +String fullName
        +String phoneNumber
        +String email
        +Address homeAddress
        +scheduleAppointment() Boolean
        +payBill(amount) Boolean
    }
    
    class MedicalRecord {
        +Date recordDate
        +String diagnosis
        +String treatment
        +String notes
        +Double cost
    }
    
    Animal <|-- Dog
    Animal <|-- Cat
    Animal --> VeterinaryRecord
    Owner --> Animal
    VeterinaryRecord --> MedicalRecord
    
    Dog : +List~Toy~ favoriteToys
    Cat : +List~String~ scratchingPosts
```

## StateDiagram Test Diagrams

### Simple Version
```mermaid
stateDiagram-v2
    [*] --> Still
    Still --> [*]
    Still --> Moving
    Moving --> Still
    Moving --> Crash
    Crash --> [*]
```

### Complex Version  
```mermaid
stateDiagram-v2
    [*] --> Idle : System Start
    
    Idle --> Loading : User Action
    Loading --> Idle : Cancel
    Loading --> Processing : Data Ready
    
    Processing --> Success : Valid Data
    Processing --> Error : Invalid Data
    Processing --> Timeout : Time Exceeded
    
    Success --> Idle : Reset
    Error --> Idle : Reset
    Error --> Retry : User Retry
    Timeout --> Retry : Auto Retry
    
    Retry --> Processing : Attempt Again
    Retry --> Failed : Max Retries Reached
    Failed --> Idle : Reset
    
    state Processing {
        [*] --> Validating
        Validating --> Transforming : Valid
        Validating --> [*] : Invalid
        Transforming --> Saving
        Saving --> [*] : Complete
    }
    
    state Error {
        [*] --> NetworkError
        [*] --> ValidationError
        [*] --> ServerError
        NetworkError --> [*]
        ValidationError --> [*]
        ServerError --> [*]
    }
```

## ErDiagram Test Diagrams

### Simple Version
```mermaid
erDiagram
    CUSTOMER ||--o{ ORDER : places
    ORDER ||--|{ LINE-ITEM : contains
    CUSTOMER }|..|{ DELIVERY-ADDRESS : uses
```

### Complex Version  
```mermaid
erDiagram
    CUSTOMER ||--o{ ORDER : "places"
    CUSTOMER {
        string customer_id PK
        string first_name
        string last_name
        string email UK
        string phone
        date created_at
        date updated_at
        boolean is_active
    }
    
    ORDER ||--|{ ORDER_ITEM : "contains"
    ORDER {
        string order_id PK
        string customer_id FK
        date order_date
        decimal total_amount
        string status
        string shipping_address
        date shipped_at
        string tracking_number
    }
    
    ORDER_ITEM }|--|| PRODUCT : "references"
    ORDER_ITEM {
        string order_item_id PK
        string order_id FK
        string product_id FK
        integer quantity
        decimal unit_price
        decimal line_total
    }
    
    PRODUCT ||--o{ PRODUCT_CATEGORY : "belongs to"
    PRODUCT {
        string product_id PK
        string name
        string description
        decimal price
        integer stock_quantity
        string sku UK
        boolean is_active
        date created_at
    }
    
    PRODUCT_CATEGORY {
        string category_id PK
        string name
        string description
        string parent_category_id FK
    }
    
    CUSTOMER }|..|{ DELIVERY_ADDRESS : "uses"
    DELIVERY_ADDRESS {
        string address_id PK
        string customer_id FK
        string street_address
        string city
        string state
        string zip_code
        string country
        boolean is_default
    }
    
    ORDER }|--|| PAYMENT : "processed by"
    PAYMENT {
        string payment_id PK
        string order_id FK
        decimal amount
        string payment_method
        string transaction_id
        date payment_date
        string status
    }
```

## Gantt Test Diagrams

### Simple Version
```mermaid
gantt
    title A Simple Gantt Diagram
    dateFormat  YYYY-MM-DD
    section Section
    A task           :a1, 2024-01-01, 30d
    Another task     :after a1  , 20d
```

### Complex Version  
```mermaid
gantt
    title Software Development Project Timeline
    dateFormat YYYY-MM-DD
    axisFormat %m/%d
    
    section Planning Phase
    Project Kickoff        :milestone, m1, 2024-01-01, 0d
    Requirements Gathering :active, req, 2024-01-02, 10d
    System Architecture    :arch, after req, 8d
    UI/UX Design          :design, 2024-01-05, 15d
    Technical Specs       :specs, after arch, 5d
    
    section Development
    Setup Development Environment :setup, after specs, 3d
    Backend API Development       :backend, after setup, 25d
    Frontend Development          :frontend, after design, 30d
    Database Design & Setup       :database, after specs, 8d
    Integration Testing           :integration, after backend, 10d
    
    section Testing & QA
    Unit Testing                 :testing, after backend, 15d
    User Acceptance Testing      :uat, after frontend, 8d
    Performance Testing          :perf, after integration, 5d
    Security Review             :security, after uat, 3d
    Bug Fixes                   :bugs, after security, 10d
    
    section Deployment
    Production Setup            :prod-setup, after bugs, 5d
    Deployment                  :deploy, after prod-setup, 2d
    Go Live                     :milestone, m2, after deploy, 0d
    Post-Launch Support         :support, after deploy, 14d
    
    section Documentation
    API Documentation           :api-docs, after backend, 8d
    User Manual                :user-docs, after uat, 10d
    Deployment Guide           :deploy-docs, after prod-setup, 5d
```

## Pie Test Diagrams

### Simple Version
```mermaid
pie title Simple Pie Chart
    "A" : 386
    "B" : 85
    "C" : 15
```

### Complex Version  
```mermaid
pie title Comprehensive Sales Analysis Q4 2024
    "Enterprise Solutions (42%)" : 420000
    "SaaS Subscriptions (28%)" : 280000
    "Professional Services (15%)" : 150000
    "Training & Certification (8%)" : 80000
    "Support & Maintenance (4%)" : 40000
    "Hardware Sales (2%)" : 20000
    "Other Revenue (1%)" : 10000
```

## Journey Test Diagrams

### Simple Version
```mermaid
journey
    title My working day
    section Go to work
      Make tea: 5: Me
      Go upstairs: 3: Me
    section Work
      Do work: 1: Me, Cat
```

### Complex Version  
```mermaid
journey
    title Complete Customer Onboarding Journey
    section Discovery
      Research Solutions    : 3: Customer
      Compare Vendors      : 2: Customer
      Request Demo         : 4: Customer, Sales
      Attend Demo          : 5: Customer, Sales, Solutions Engineer
      
    section Evaluation
      Technical Review     : 3: Customer, IT Team
      Security Assessment  : 2: Customer, Security Team
      Cost Analysis        : 4: Customer, Finance
      Reference Calls      : 5: Customer, Sales
      
    section Purchase
      Contract Negotiation : 2: Customer, Sales, Legal
      Approval Process     : 3: Customer, Management
      Contract Signing     : 5: Customer, Sales
      
    section Implementation  
      Project Kickoff      : 5: Customer, Success Manager, Implementation
      System Configuration: 3: Customer, Implementation, IT Team
      Data Migration       : 2: Customer, Implementation, Data Team
      Integration Setup    : 3: Customer, Implementation, IT Team
      Testing Phase        : 4: Customer, Implementation, QA
      
    section Go-Live
      User Training        : 4: Customer, Success Manager, Trainer
      Soft Launch         : 3: Customer, Success Manager
      Production Rollout   : 5: Customer, Success Manager, Support
      Post-Launch Review   : 4: Customer, Success Manager
```

## Gitgraph Test Diagrams

### Simple Version
```mermaid
gitgraph
    commit
    commit
    branch develop
    commit
    commit
    checkout main
    commit
    merge develop
```

### Complex Version  
```mermaid
gitgraph
    commit id: "Initial commit"
    commit id: "Add basic structure"
    
    branch feature/authentication
    checkout feature/authentication
    commit id: "Add login form"
    commit id: "Implement JWT auth"
    commit id: "Add password validation"
    
    checkout main
    commit id: "Update documentation"
    
    branch feature/user-management
    checkout feature/user-management  
    commit id: "Add user model"
    commit id: "Create user CRUD"
    
    checkout main
    merge feature/authentication
    commit id: "Version 1.1.0" tag: "v1.1.0"
    
    checkout feature/user-management
    commit id: "Add user permissions"
    commit id: "Implement user roles"
    
    checkout main
    branch hotfix/security-patch
    commit id: "Fix SQL injection"
    commit id: "Update dependencies"
    
    checkout main  
    merge hotfix/security-patch
    commit id: "Security patch 1.1.1" tag: "v1.1.1"
    
    merge feature/user-management
    commit id: "Version 1.2.0" tag: "v1.2.0"
    
    branch feature/dashboard
    checkout feature/dashboard
    commit id: "Add dashboard layout"
    commit id: "Implement analytics"
```

## Mindmap Test Diagrams

### Simple Version
```mermaid
mindmap
  root((mindmap))
    Origins
      Long history
    Research
      On effectiveness
```

### Complex Version  
```mermaid
mindmap
  root((Software Architecture))
    Frontend
      Frameworks
        React
          Components
          State Management
            Redux
            Context API
          Hooks
        Vue.js
          Composition API
          Single File Components
        Angular
          TypeScript
          Dependency Injection
      Styling
        CSS Frameworks
          Bootstrap
          Tailwind CSS
        Preprocessors
          SASS
          LESS
        CSS-in-JS
          Styled Components
          Emotion
    Backend
      Languages
        JavaScript
          Node.js
            Express
            FastAPI
        Python
          Django
          Flask
          FastAPI
        Java
          Spring Boot
          Microservices
        Go
          Gin
          Fiber
      Databases
        SQL
          PostgreSQL
          MySQL
        NoSQL
          MongoDB
          Redis
          Elasticsearch
    DevOps
      Containerization
        Docker
          Images
          Containers
          Docker Compose
        Kubernetes
          Pods
          Services
          Ingress
      CI/CD
        GitHub Actions
        Jenkins
        GitLab CI
      Cloud Providers
        AWS
          EC2
          S3
          Lambda
        Azure
          App Service
          Blob Storage
        Google Cloud
          Compute Engine
          Cloud Storage
```

**Export Options**:
- Formats: SVG, PNG, JPG
- Theme: default
- Dimensions: 800x600
- Background: transparent

## Results Summary

## Flowchart Results

### CLI (SVG) (simple) Strategy

- **Status**: ❌ FAILED
- **Duration**: 37ms
- **Version**: N/A
- **Error**: CLI not available - mmdc command not found. Install with: npm install -g @mermaid-js/mermaid-cli

### Web (SVG) (simple) Strategy

- **Status**: ✅ SUCCESS
- **Duration**: 747ms
- **Version**: N/A
- **Output Files**:
  - diagram.svg (14428 bytes)

### CLI (PNG) (simple) Strategy

- **Status**: ❌ FAILED
- **Duration**: 42ms
- **Version**: N/A
- **Error**: CLI not available - mmdc command not found. Install with: npm install -g @mermaid-js/mermaid-cli

### Web (PNG) (simple) Strategy

- **Status**: ✅ SUCCESS
- **Duration**: 554ms
- **Version**: N/A
- **Output Files**:
  - diagram.png (26814 bytes)

### CLI (JPG) (simple) Strategy

- **Status**: ❌ FAILED
- **Duration**: 42ms
- **Version**: N/A
- **Error**: CLI not available - mmdc command not found. Install with: npm install -g @mermaid-js/mermaid-cli

### Web (JPG) (simple) Strategy

- **Status**: ✅ SUCCESS
- **Duration**: 330ms
- **Version**: N/A
- **Output Files**:
  - diagram.jpg (20893 bytes)

### CLI (SVG) (complex) Strategy

- **Status**: ❌ FAILED
- **Duration**: 40ms
- **Version**: N/A
- **Error**: CLI not available - mmdc command not found. Install with: npm install -g @mermaid-js/mermaid-cli

### Web (SVG) (complex) Strategy

- **Status**: ✅ SUCCESS
- **Duration**: 370ms
- **Version**: N/A
- **Output Files**:
  - diagram.svg (70895 bytes)

### CLI (PNG) (complex) Strategy

- **Status**: ❌ FAILED
- **Duration**: 47ms
- **Version**: N/A
- **Error**: CLI not available - mmdc command not found. Install with: npm install -g @mermaid-js/mermaid-cli

### Web (PNG) (complex) Strategy

- **Status**: ✅ SUCCESS
- **Duration**: 469ms
- **Version**: N/A
- **Output Files**:
  - diagram.png (34736 bytes)

### CLI (JPG) (complex) Strategy

- **Status**: ❌ FAILED
- **Duration**: 49ms
- **Version**: N/A
- **Error**: CLI not available - mmdc command not found. Install with: npm install -g @mermaid-js/mermaid-cli

### Web (JPG) (complex) Strategy

- **Status**: ✅ SUCCESS
- **Duration**: 626ms
- **Version**: N/A
- **Output Files**:
  - diagram.jpg (18908 bytes)

## Sequence Results

### CLI (SVG) (simple) Strategy

- **Status**: ❌ FAILED
- **Duration**: 76ms
- **Version**: N/A
- **Error**: CLI not available - mmdc command not found. Install with: npm install -g @mermaid-js/mermaid-cli

### Web (SVG) (simple) Strategy

- **Status**: ✅ SUCCESS
- **Duration**: 272ms
- **Version**: N/A
- **Output Files**:
  - diagram.svg (21507 bytes)

### CLI (PNG) (simple) Strategy

- **Status**: ❌ FAILED
- **Duration**: 47ms
- **Version**: N/A
- **Error**: CLI not available - mmdc command not found. Install with: npm install -g @mermaid-js/mermaid-cli

### Web (PNG) (simple) Strategy

- **Status**: ✅ SUCCESS
- **Duration**: 302ms
- **Version**: N/A
- **Output Files**:
  - diagram.png (31578 bytes)

### CLI (JPG) (simple) Strategy

- **Status**: ❌ FAILED
- **Duration**: 46ms
- **Version**: N/A
- **Error**: CLI not available - mmdc command not found. Install with: npm install -g @mermaid-js/mermaid-cli

### Web (JPG) (simple) Strategy

- **Status**: ✅ SUCCESS
- **Duration**: 304ms
- **Version**: N/A
- **Output Files**:
  - diagram.jpg (21105 bytes)

### CLI (SVG) (complex) Strategy

- **Status**: ❌ FAILED
- **Duration**: 54ms
- **Version**: N/A
- **Error**: CLI not available - mmdc command not found. Install with: npm install -g @mermaid-js/mermaid-cli

### Web (SVG) (complex) Strategy

- **Status**: ✅ SUCCESS
- **Duration**: 462ms
- **Version**: N/A
- **Output Files**:
  - diagram.svg (33709 bytes)

### CLI (PNG) (complex) Strategy

- **Status**: ❌ FAILED
- **Duration**: 72ms
- **Version**: N/A
- **Error**: CLI not available - mmdc command not found. Install with: npm install -g @mermaid-js/mermaid-cli

### Web (PNG) (complex) Strategy

- **Status**: ✅ SUCCESS
- **Duration**: 344ms
- **Version**: N/A
- **Output Files**:
  - diagram.png (59280 bytes)

### CLI (JPG) (complex) Strategy

- **Status**: ❌ FAILED
- **Duration**: 51ms
- **Version**: N/A
- **Error**: CLI not available - mmdc command not found. Install with: npm install -g @mermaid-js/mermaid-cli

### Web (JPG) (complex) Strategy

- **Status**: ✅ SUCCESS
- **Duration**: 297ms
- **Version**: N/A
- **Output Files**:
  - diagram.jpg (35270 bytes)

## ClassDiagram Results

### CLI (SVG) (simple) Strategy

- **Status**: ❌ FAILED
- **Duration**: 38ms
- **Version**: N/A
- **Error**: CLI not available - mmdc command not found. Install with: npm install -g @mermaid-js/mermaid-cli

### Web (SVG) (simple) Strategy

- **Status**: ✅ SUCCESS
- **Duration**: 485ms
- **Version**: N/A
- **Output Files**:
  - diagram.svg (14924 bytes)

### CLI (PNG) (simple) Strategy

- **Status**: ❌ FAILED
- **Duration**: 39ms
- **Version**: N/A
- **Error**: CLI not available - mmdc command not found. Install with: npm install -g @mermaid-js/mermaid-cli

### Web (PNG) (simple) Strategy

- **Status**: ✅ SUCCESS
- **Duration**: 317ms
- **Version**: N/A
- **Output Files**:
  - diagram.png (28979 bytes)

### CLI (JPG) (simple) Strategy

- **Status**: ❌ FAILED
- **Duration**: 41ms
- **Version**: N/A
- **Error**: CLI not available - mmdc command not found. Install with: npm install -g @mermaid-js/mermaid-cli

### Web (JPG) (simple) Strategy

- **Status**: ✅ SUCCESS
- **Duration**: 313ms
- **Version**: N/A
- **Output Files**:
  - diagram.jpg (21712 bytes)

### CLI (SVG) (complex) Strategy

- **Status**: ❌ FAILED
- **Duration**: 39ms
- **Version**: N/A
- **Error**: CLI not available - mmdc command not found. Install with: npm install -g @mermaid-js/mermaid-cli

### Web (SVG) (complex) Strategy

- **Status**: ✅ SUCCESS
- **Duration**: 546ms
- **Version**: N/A
- **Output Files**:
  - diagram.svg (42065 bytes)

### CLI (PNG) (complex) Strategy

- **Status**: ❌ FAILED
- **Duration**: 59ms
- **Version**: N/A
- **Error**: CLI not available - mmdc command not found. Install with: npm install -g @mermaid-js/mermaid-cli

### Web (PNG) (complex) Strategy

- **Status**: ✅ SUCCESS
- **Duration**: 516ms
- **Version**: N/A
- **Output Files**:
  - diagram.png (71186 bytes)

### CLI (JPG) (complex) Strategy

- **Status**: ❌ FAILED
- **Duration**: 43ms
- **Version**: N/A
- **Error**: CLI not available - mmdc command not found. Install with: npm install -g @mermaid-js/mermaid-cli

### Web (JPG) (complex) Strategy

- **Status**: ✅ SUCCESS
- **Duration**: 370ms
- **Version**: N/A
- **Output Files**:
  - diagram.jpg (47057 bytes)

## StateDiagram Results

### CLI (SVG) (simple) Strategy

- **Status**: ❌ FAILED
- **Duration**: 38ms
- **Version**: N/A
- **Error**: CLI not available - mmdc command not found. Install with: npm install -g @mermaid-js/mermaid-cli

### Web (SVG) (simple) Strategy

- **Status**: ✅ SUCCESS
- **Duration**: 517ms
- **Version**: N/A
- **Output Files**:
  - diagram.svg (137535 bytes)

### CLI (PNG) (simple) Strategy

- **Status**: ❌ FAILED
- **Duration**: 44ms
- **Version**: N/A
- **Error**: CLI not available - mmdc command not found. Install with: npm install -g @mermaid-js/mermaid-cli

### Web (PNG) (simple) Strategy

- **Status**: ✅ SUCCESS
- **Duration**: 345ms
- **Version**: N/A
- **Output Files**:
  - diagram.png (26510 bytes)

### CLI (JPG) (simple) Strategy

- **Status**: ❌ FAILED
- **Duration**: 41ms
- **Version**: N/A
- **Error**: CLI not available - mmdc command not found. Install with: npm install -g @mermaid-js/mermaid-cli

### Web (JPG) (simple) Strategy

- **Status**: ✅ SUCCESS
- **Duration**: 388ms
- **Version**: N/A
- **Output Files**:
  - diagram.jpg (16952 bytes)

### CLI (SVG) (complex) Strategy

- **Status**: ❌ FAILED
- **Duration**: 46ms
- **Version**: N/A
- **Error**: CLI not available - mmdc command not found. Install with: npm install -g @mermaid-js/mermaid-cli

### Web (SVG) (complex) Strategy

- **Status**: ✅ SUCCESS
- **Duration**: 499ms
- **Version**: N/A
- **Output Files**:
  - diagram.svg (493643 bytes)

### CLI (PNG) (complex) Strategy

- **Status**: ❌ FAILED
- **Duration**: 53ms
- **Version**: N/A
- **Error**: CLI not available - mmdc command not found. Install with: npm install -g @mermaid-js/mermaid-cli

### Web (PNG) (complex) Strategy

- **Status**: ✅ SUCCESS
- **Duration**: 631ms
- **Version**: N/A
- **Output Files**:
  - diagram.png (65642 bytes)

### CLI (JPG) (complex) Strategy

- **Status**: ❌ FAILED
- **Duration**: 42ms
- **Version**: N/A
- **Error**: CLI not available - mmdc command not found. Install with: npm install -g @mermaid-js/mermaid-cli

### Web (JPG) (complex) Strategy

- **Status**: ✅ SUCCESS
- **Duration**: 468ms
- **Version**: N/A
- **Output Files**:
  - diagram.jpg (28626 bytes)

## ErDiagram Results

### CLI (SVG) (simple) Strategy

- **Status**: ❌ FAILED
- **Duration**: 97ms
- **Version**: N/A
- **Error**: CLI not available - mmdc command not found. Install with: npm install -g @mermaid-js/mermaid-cli

### Web (SVG) (simple) Strategy

- **Status**: ✅ SUCCESS
- **Duration**: 342ms
- **Version**: N/A
- **Output Files**:
  - diagram.svg (9929 bytes)

### CLI (PNG) (simple) Strategy

- **Status**: ❌ FAILED
- **Duration**: 47ms
- **Version**: N/A
- **Error**: CLI not available - mmdc command not found. Install with: npm install -g @mermaid-js/mermaid-cli

### Web (PNG) (simple) Strategy

- **Status**: ✅ SUCCESS
- **Duration**: 378ms
- **Version**: N/A
- **Output Files**:
  - diagram.png (31094 bytes)

### CLI (JPG) (simple) Strategy

- **Status**: ❌ FAILED
- **Duration**: 49ms
- **Version**: N/A
- **Error**: CLI not available - mmdc command not found. Install with: npm install -g @mermaid-js/mermaid-cli

### Web (JPG) (simple) Strategy

- **Status**: ✅ SUCCESS
- **Duration**: 328ms
- **Version**: N/A
- **Output Files**:
  - diagram.jpg (21381 bytes)

### CLI (SVG) (complex) Strategy

- **Status**: ❌ FAILED
- **Duration**: 48ms
- **Version**: N/A
- **Error**: CLI not available - mmdc command not found. Install with: npm install -g @mermaid-js/mermaid-cli

### Web (SVG) (complex) Strategy

- **Status**: ✅ SUCCESS
- **Duration**: 566ms
- **Version**: N/A
- **Output Files**:
  - diagram.svg (158636 bytes)

### CLI (PNG) (complex) Strategy

- **Status**: ❌ FAILED
- **Duration**: 53ms
- **Version**: N/A
- **Error**: CLI not available - mmdc command not found. Install with: npm install -g @mermaid-js/mermaid-cli

### Web (PNG) (complex) Strategy

- **Status**: ✅ SUCCESS
- **Duration**: 859ms
- **Version**: N/A
- **Output Files**:
  - diagram.png (63981 bytes)

### CLI (JPG) (complex) Strategy

- **Status**: ❌ FAILED
- **Duration**: 41ms
- **Version**: N/A
- **Error**: CLI not available - mmdc command not found. Install with: npm install -g @mermaid-js/mermaid-cli

### Web (JPG) (complex) Strategy

- **Status**: ✅ SUCCESS
- **Duration**: 659ms
- **Version**: N/A
- **Output Files**:
  - diagram.jpg (32174 bytes)

## Gantt Results

### CLI (SVG) (simple) Strategy

- **Status**: ❌ FAILED
- **Duration**: 47ms
- **Version**: N/A
- **Error**: CLI not available - mmdc command not found. Install with: npm install -g @mermaid-js/mermaid-cli

### Web (SVG) (simple) Strategy

- **Status**: ✅ SUCCESS
- **Duration**: 244ms
- **Version**: N/A
- **Output Files**:
  - diagram.svg (10160 bytes)

### CLI (PNG) (simple) Strategy

- **Status**: ❌ FAILED
- **Duration**: 44ms
- **Version**: N/A
- **Error**: CLI not available - mmdc command not found. Install with: npm install -g @mermaid-js/mermaid-cli

### Web (PNG) (simple) Strategy

- **Status**: ✅ SUCCESS
- **Duration**: 303ms
- **Version**: N/A
- **Output Files**:
  - diagram.png (20661 bytes)

### CLI (JPG) (simple) Strategy

- **Status**: ❌ FAILED
- **Duration**: 47ms
- **Version**: N/A
- **Error**: CLI not available - mmdc command not found. Install with: npm install -g @mermaid-js/mermaid-cli

### Web (JPG) (simple) Strategy

- **Status**: ✅ SUCCESS
- **Duration**: 304ms
- **Version**: N/A
- **Output Files**:
  - diagram.jpg (17852 bytes)

### CLI (SVG) (complex) Strategy

- **Status**: ❌ FAILED
- **Duration**: 48ms
- **Version**: N/A
- **Error**: CLI not available - mmdc command not found. Install with: npm install -g @mermaid-js/mermaid-cli

### Web (SVG) (complex) Strategy

- **Status**: ✅ SUCCESS
- **Duration**: 264ms
- **Version**: N/A
- **Output Files**:
  - diagram.svg (19773 bytes)

### CLI (PNG) (complex) Strategy

- **Status**: ❌ FAILED
- **Duration**: 49ms
- **Version**: N/A
- **Error**: CLI not available - mmdc command not found. Install with: npm install -g @mermaid-js/mermaid-cli

### Web (PNG) (complex) Strategy

- **Status**: ✅ SUCCESS
- **Duration**: 331ms
- **Version**: N/A
- **Output Files**:
  - diagram.png (70026 bytes)

### CLI (JPG) (complex) Strategy

- **Status**: ❌ FAILED
- **Duration**: 45ms
- **Version**: N/A
- **Error**: CLI not available - mmdc command not found. Install with: npm install -g @mermaid-js/mermaid-cli

### Web (JPG) (complex) Strategy

- **Status**: ✅ SUCCESS
- **Duration**: 294ms
- **Version**: N/A
- **Output Files**:
  - diagram.jpg (59670 bytes)

## Pie Results

### CLI (SVG) (simple) Strategy

- **Status**: ❌ FAILED
- **Duration**: 43ms
- **Version**: N/A
- **Error**: CLI not available - mmdc command not found. Install with: npm install -g @mermaid-js/mermaid-cli

### Web (SVG) (simple) Strategy

- **Status**: ✅ SUCCESS
- **Duration**: 260ms
- **Version**: N/A
- **Output Files**:
  - diagram.svg (3882 bytes)

### CLI (PNG) (simple) Strategy

- **Status**: ❌ FAILED
- **Duration**: 46ms
- **Version**: N/A
- **Error**: CLI not available - mmdc command not found. Install with: npm install -g @mermaid-js/mermaid-cli

### Web (PNG) (simple) Strategy

- **Status**: ✅ SUCCESS
- **Duration**: 293ms
- **Version**: N/A
- **Output Files**:
  - diagram.png (49320 bytes)

### CLI (JPG) (simple) Strategy

- **Status**: ❌ FAILED
- **Duration**: 96ms
- **Version**: N/A
- **Error**: CLI not available - mmdc command not found. Install with: npm install -g @mermaid-js/mermaid-cli

### Web (JPG) (simple) Strategy

- **Status**: ✅ SUCCESS
- **Duration**: 359ms
- **Version**: N/A
- **Output Files**:
  - diagram.jpg (27470 bytes)

### CLI (SVG) (complex) Strategy

- **Status**: ❌ FAILED
- **Duration**: 54ms
- **Version**: N/A
- **Error**: CLI not available - mmdc command not found. Install with: npm install -g @mermaid-js/mermaid-cli

### Web (SVG) (complex) Strategy

- **Status**: ✅ SUCCESS
- **Duration**: 313ms
- **Version**: N/A
- **Output Files**:
  - diagram.svg (5766 bytes)

### CLI (PNG) (complex) Strategy

- **Status**: ❌ FAILED
- **Duration**: 40ms
- **Version**: N/A
- **Error**: CLI not available - mmdc command not found. Install with: npm install -g @mermaid-js/mermaid-cli

### Web (PNG) (complex) Strategy

- **Status**: ✅ SUCCESS
- **Duration**: 324ms
- **Version**: N/A
- **Output Files**:
  - diagram.png (75266 bytes)

### CLI (JPG) (complex) Strategy

- **Status**: ❌ FAILED
- **Duration**: 41ms
- **Version**: N/A
- **Error**: CLI not available - mmdc command not found. Install with: npm install -g @mermaid-js/mermaid-cli

### Web (JPG) (complex) Strategy

- **Status**: ✅ SUCCESS
- **Duration**: 320ms
- **Version**: N/A
- **Output Files**:
  - diagram.jpg (52506 bytes)

## Journey Results

### CLI (SVG) (simple) Strategy

- **Status**: ❌ FAILED
- **Duration**: 45ms
- **Version**: N/A
- **Error**: CLI not available - mmdc command not found. Install with: npm install -g @mermaid-js/mermaid-cli

### Web (SVG) (simple) Strategy

- **Status**: ✅ SUCCESS
- **Duration**: 448ms
- **Version**: N/A
- **Output Files**:
  - diagram.svg (11263 bytes)

### CLI (PNG) (simple) Strategy

- **Status**: ❌ FAILED
- **Duration**: 42ms
- **Version**: N/A
- **Error**: CLI not available - mmdc command not found. Install with: npm install -g @mermaid-js/mermaid-cli

### Web (PNG) (simple) Strategy

- **Status**: ✅ SUCCESS
- **Duration**: 353ms
- **Version**: N/A
- **Output Files**:
  - diagram.png (34950 bytes)

### CLI (JPG) (simple) Strategy

- **Status**: ❌ FAILED
- **Duration**: 66ms
- **Version**: N/A
- **Error**: CLI not available - mmdc command not found. Install with: npm install -g @mermaid-js/mermaid-cli

### Web (JPG) (simple) Strategy

- **Status**: ✅ SUCCESS
- **Duration**: 267ms
- **Version**: N/A
- **Output Files**:
  - diagram.jpg (22947 bytes)

### CLI (SVG) (complex) Strategy

- **Status**: ❌ FAILED
- **Duration**: 44ms
- **Version**: N/A
- **Error**: CLI not available - mmdc command not found. Install with: npm install -g @mermaid-js/mermaid-cli

### Web (SVG) (complex) Strategy

- **Status**: ✅ SUCCESS
- **Duration**: 237ms
- **Version**: N/A
- **Output Files**:
  - diagram.svg (45179 bytes)

### CLI (PNG) (complex) Strategy

- **Status**: ❌ FAILED
- **Duration**: 45ms
- **Version**: N/A
- **Error**: CLI not available - mmdc command not found. Install with: npm install -g @mermaid-js/mermaid-cli

### Web (PNG) (complex) Strategy

- **Status**: ✅ SUCCESS
- **Duration**: 351ms
- **Version**: N/A
- **Output Files**:
  - diagram.png (28277 bytes)

### CLI (JPG) (complex) Strategy

- **Status**: ❌ FAILED
- **Duration**: 71ms
- **Version**: N/A
- **Error**: CLI not available - mmdc command not found. Install with: npm install -g @mermaid-js/mermaid-cli

### Web (JPG) (complex) Strategy

- **Status**: ✅ SUCCESS
- **Duration**: 303ms
- **Version**: N/A
- **Output Files**:
  - diagram.jpg (15573 bytes)

## Gitgraph Results

### CLI (SVG) (simple) Strategy

- **Status**: ❌ FAILED
- **Duration**: 40ms
- **Version**: N/A
- **Error**: CLI not available - mmdc command not found. Install with: npm install -g @mermaid-js/mermaid-cli

### Web (SVG) (simple) Strategy

- **Status**: ❌ FAILED
- **Duration**: 260ms
- **Version**: N/A
- **Error**: Web export strategy failed: Webview rendering error: No diagram type detected matching given configuration for text: gitgraph
    commit
    commit
    branch develop
    commit
    commit
    checkout main
    commit
    merge develop\nUnknownDiagramError: No diagram type detected matching given configuration for text: gitgraph
    commit
    commit
    branch develop
    commit
    commit
    checkout main
    commit
    merge develop
    at detectType (https://file+.vscode-resource.vscode-cdn.net/c%3A/Users/bibol/.vscode/extensions/gsejas.mermaid-export-pro-1.0.9/dist/webview.js:263:74)
    at Diagram.fromText (https://file+.vscode-resource.vscode-cdn.net/c%3A/Users/bibol/.vscode/extensions/gsejas.mermaid-export-pro-1.0.9/dist/webview.js:2728:6242)
    at Object.render (https://file+.vscode-resource.vscode-cdn.net/c%3A/Users/bibol/.vscode/extensions/gsejas.mermaid-export-pro-1.0.9/dist/webview.js:2736:1192)
    at https://file+.vscode-resource.vscode-cdn.net/c%3A/Users/bibol/.vscode/extensions/gsejas.mermaid-export-pro-1.0.9/dist/webview.js:2736:5557
    at new Promise (<anonymous>)
    at performCall (https://file+.vscode-resource.vscode-cdn.net/c%3A/Users/bibol/.vscode/extensions/gsejas.mermaid-export-pro-1.0.9/dist/webview.js:2736:5534)
    at executeQueue (https://file+.vscode-resource.vscode-cdn.net/c%3A/Users/bibol/.vscode/extensions/gsejas.mermaid-export-pro-1.0.9/dist/webview.js:2736:5192)
    at https://file+.vscode-resource.vscode-cdn.net/c%3A/Users/bibol/.vscode/extensions/gsejas.mermaid-export-pro-1.0.9/dist/webview.js:2736:5682
    at new Promise (<anonymous>)
    at Object.render (https://file+.vscode-resource.vscode-cdn.net/c%3A/Users/bibol/.vscode/extensions/gsejas.mermaid-export-pro-1.0.9/dist/webview.js:2736:5502)

### CLI (PNG) (simple) Strategy

- **Status**: ❌ FAILED
- **Duration**: 46ms
- **Version**: N/A
- **Error**: CLI not available - mmdc command not found. Install with: npm install -g @mermaid-js/mermaid-cli

### Web (PNG) (simple) Strategy

- **Status**: ❌ FAILED
- **Duration**: 237ms
- **Version**: N/A
- **Error**: Web export strategy failed: Webview rendering error: No diagram type detected matching given configuration for text: gitgraph
    commit
    commit
    branch develop
    commit
    commit
    checkout main
    commit
    merge develop\nUnknownDiagramError: No diagram type detected matching given configuration for text: gitgraph
    commit
    commit
    branch develop
    commit
    commit
    checkout main
    commit
    merge develop
    at detectType (https://file+.vscode-resource.vscode-cdn.net/c%3A/Users/bibol/.vscode/extensions/gsejas.mermaid-export-pro-1.0.9/dist/webview.js:263:74)
    at Diagram.fromText (https://file+.vscode-resource.vscode-cdn.net/c%3A/Users/bibol/.vscode/extensions/gsejas.mermaid-export-pro-1.0.9/dist/webview.js:2728:6242)
    at Object.render (https://file+.vscode-resource.vscode-cdn.net/c%3A/Users/bibol/.vscode/extensions/gsejas.mermaid-export-pro-1.0.9/dist/webview.js:2736:1192)
    at https://file+.vscode-resource.vscode-cdn.net/c%3A/Users/bibol/.vscode/extensions/gsejas.mermaid-export-pro-1.0.9/dist/webview.js:2736:5557
    at new Promise (<anonymous>)
    at performCall (https://file+.vscode-resource.vscode-cdn.net/c%3A/Users/bibol/.vscode/extensions/gsejas.mermaid-export-pro-1.0.9/dist/webview.js:2736:5534)
    at executeQueue (https://file+.vscode-resource.vscode-cdn.net/c%3A/Users/bibol/.vscode/extensions/gsejas.mermaid-export-pro-1.0.9/dist/webview.js:2736:5192)
    at https://file+.vscode-resource.vscode-cdn.net/c%3A/Users/bibol/.vscode/extensions/gsejas.mermaid-export-pro-1.0.9/dist/webview.js:2736:5682
    at new Promise (<anonymous>)
    at Object.render (https://file+.vscode-resource.vscode-cdn.net/c%3A/Users/bibol/.vscode/extensions/gsejas.mermaid-export-pro-1.0.9/dist/webview.js:2736:5502)

### CLI (JPG) (simple) Strategy

- **Status**: ❌ FAILED
- **Duration**: 41ms
- **Version**: N/A
- **Error**: CLI not available - mmdc command not found. Install with: npm install -g @mermaid-js/mermaid-cli

### Web (JPG) (simple) Strategy

- **Status**: ❌ FAILED
- **Duration**: 490ms
- **Version**: N/A
- **Error**: Web export strategy failed: Webview rendering error: No diagram type detected matching given configuration for text: gitgraph
    commit
    commit
    branch develop
    commit
    commit
    checkout main
    commit
    merge develop\nUnknownDiagramError: No diagram type detected matching given configuration for text: gitgraph
    commit
    commit
    branch develop
    commit
    commit
    checkout main
    commit
    merge develop
    at detectType (https://file+.vscode-resource.vscode-cdn.net/c%3A/Users/bibol/.vscode/extensions/gsejas.mermaid-export-pro-1.0.9/dist/webview.js:263:74)
    at Diagram.fromText (https://file+.vscode-resource.vscode-cdn.net/c%3A/Users/bibol/.vscode/extensions/gsejas.mermaid-export-pro-1.0.9/dist/webview.js:2728:6242)
    at Object.render (https://file+.vscode-resource.vscode-cdn.net/c%3A/Users/bibol/.vscode/extensions/gsejas.mermaid-export-pro-1.0.9/dist/webview.js:2736:1192)
    at https://file+.vscode-resource.vscode-cdn.net/c%3A/Users/bibol/.vscode/extensions/gsejas.mermaid-export-pro-1.0.9/dist/webview.js:2736:5557
    at new Promise (<anonymous>)
    at performCall (https://file+.vscode-resource.vscode-cdn.net/c%3A/Users/bibol/.vscode/extensions/gsejas.mermaid-export-pro-1.0.9/dist/webview.js:2736:5534)
    at executeQueue (https://file+.vscode-resource.vscode-cdn.net/c%3A/Users/bibol/.vscode/extensions/gsejas.mermaid-export-pro-1.0.9/dist/webview.js:2736:5192)
    at https://file+.vscode-resource.vscode-cdn.net/c%3A/Users/bibol/.vscode/extensions/gsejas.mermaid-export-pro-1.0.9/dist/webview.js:2736:5682
    at new Promise (<anonymous>)
    at Object.render (https://file+.vscode-resource.vscode-cdn.net/c%3A/Users/bibol/.vscode/extensions/gsejas.mermaid-export-pro-1.0.9/dist/webview.js:2736:5502)

### CLI (SVG) (complex) Strategy

- **Status**: ❌ FAILED
- **Duration**: 58ms
- **Version**: N/A
- **Error**: CLI not available - mmdc command not found. Install with: npm install -g @mermaid-js/mermaid-cli

### Web (SVG) (complex) Strategy

- **Status**: ❌ FAILED
- **Duration**: 290ms
- **Version**: N/A
- **Error**: Web export strategy failed: Webview rendering error: No diagram type detected matching given configuration for text: gitgraph
    commit id: "Initial commit"
    commit id: "Add basic structure"
    
    branch feature/authentication
    checkout feature/authentication
    commit id: "Add login form"
    commit id: "Implement JWT auth"
    commit id: "Add password validation"
    
    checkout main
    commit id: "Update documentation"
    
    branch feature/user-management
    checkout feature/user-management  
    commit id: "Add user model"
    commit id: "Create user CRUD"
    
    checkout main
    merge feature/authentication
    commit id: "Version 1.1.0" tag: "v1.1.0"
    
    checkout feature/user-management
    commit id: "Add user permissions"
    commit id: "Implement user roles"
    
    checkout main
    branch hotfix/security-patch
    commit id: "Fix SQL injection"
    commit id: "Update dependencies"
    
    checkout main  
    merge hotfix/security-patch
    commit id: "Security patch 1.1.1" tag: "v1.1.1"
    
    merge feature/user-management
    commit id: "Version 1.2.0" tag: "v1.2.0"
    
    branch feature/dashboard
    checkout feature/dashboard
    commit id: "Add dashboard layout"
    commit id: "Implement analytics"\nUnknownDiagramError: No diagram type detected matching given configuration for text: gitgraph
    commit id: "Initial commit"
    commit id: "Add basic structure"
    
    branch feature/authentication
    checkout feature/authentication
    commit id: "Add login form"
    commit id: "Implement JWT auth"
    commit id: "Add password validation"
    
    checkout main
    commit id: "Update documentation"
    
    branch feature/user-management
    checkout feature/user-management  
    commit id: "Add user model"
    commit id: "Create user CRUD"
    
    checkout main
    merge feature/authentication
    commit id: "Version 1.1.0" tag: "v1.1.0"
    
    checkout feature/user-management
    commit id: "Add user permissions"
    commit id: "Implement user roles"
    
    checkout main
    branch hotfix/security-patch
    commit id: "Fix SQL injection"
    commit id: "Update dependencies"
    
    checkout main  
    merge hotfix/security-patch
    commit id: "Security patch 1.1.1" tag: "v1.1.1"
    
    merge feature/user-management
    commit id: "Version 1.2.0" tag: "v1.2.0"
    
    branch feature/dashboard
    checkout feature/dashboard
    commit id: "Add dashboard layout"
    commit id: "Implement analytics"
    at detectType (https://file+.vscode-resource.vscode-cdn.net/c%3A/Users/bibol/.vscode/extensions/gsejas.mermaid-export-pro-1.0.9/dist/webview.js:263:74)
    at Diagram.fromText (https://file+.vscode-resource.vscode-cdn.net/c%3A/Users/bibol/.vscode/extensions/gsejas.mermaid-export-pro-1.0.9/dist/webview.js:2728:6242)
    at Object.render (https://file+.vscode-resource.vscode-cdn.net/c%3A/Users/bibol/.vscode/extensions/gsejas.mermaid-export-pro-1.0.9/dist/webview.js:2736:1192)
    at https://file+.vscode-resource.vscode-cdn.net/c%3A/Users/bibol/.vscode/extensions/gsejas.mermaid-export-pro-1.0.9/dist/webview.js:2736:5557
    at new Promise (<anonymous>)
    at performCall (https://file+.vscode-resource.vscode-cdn.net/c%3A/Users/bibol/.vscode/extensions/gsejas.mermaid-export-pro-1.0.9/dist/webview.js:2736:5534)
    at executeQueue (https://file+.vscode-resource.vscode-cdn.net/c%3A/Users/bibol/.vscode/extensions/gsejas.mermaid-export-pro-1.0.9/dist/webview.js:2736:5192)
    at https://file+.vscode-resource.vscode-cdn.net/c%3A/Users/bibol/.vscode/extensions/gsejas.mermaid-export-pro-1.0.9/dist/webview.js:2736:5682
    at new Promise (<anonymous>)
    at Object.render (https://file+.vscode-resource.vscode-cdn.net/c%3A/Users/bibol/.vscode/extensions/gsejas.mermaid-export-pro-1.0.9/dist/webview.js:2736:5502)

### CLI (PNG) (complex) Strategy

- **Status**: ❌ FAILED
- **Duration**: 48ms
- **Version**: N/A
- **Error**: CLI not available - mmdc command not found. Install with: npm install -g @mermaid-js/mermaid-cli

### Web (PNG) (complex) Strategy

- **Status**: ❌ FAILED
- **Duration**: 258ms
- **Version**: N/A
- **Error**: Web export strategy failed: Webview rendering error: No diagram type detected matching given configuration for text: gitgraph
    commit id: "Initial commit"
    commit id: "Add basic structure"
    
    branch feature/authentication
    checkout feature/authentication
    commit id: "Add login form"
    commit id: "Implement JWT auth"
    commit id: "Add password validation"
    
    checkout main
    commit id: "Update documentation"
    
    branch feature/user-management
    checkout feature/user-management  
    commit id: "Add user model"
    commit id: "Create user CRUD"
    
    checkout main
    merge feature/authentication
    commit id: "Version 1.1.0" tag: "v1.1.0"
    
    checkout feature/user-management
    commit id: "Add user permissions"
    commit id: "Implement user roles"
    
    checkout main
    branch hotfix/security-patch
    commit id: "Fix SQL injection"
    commit id: "Update dependencies"
    
    checkout main  
    merge hotfix/security-patch
    commit id: "Security patch 1.1.1" tag: "v1.1.1"
    
    merge feature/user-management
    commit id: "Version 1.2.0" tag: "v1.2.0"
    
    branch feature/dashboard
    checkout feature/dashboard
    commit id: "Add dashboard layout"
    commit id: "Implement analytics"\nUnknownDiagramError: No diagram type detected matching given configuration for text: gitgraph
    commit id: "Initial commit"
    commit id: "Add basic structure"
    
    branch feature/authentication
    checkout feature/authentication
    commit id: "Add login form"
    commit id: "Implement JWT auth"
    commit id: "Add password validation"
    
    checkout main
    commit id: "Update documentation"
    
    branch feature/user-management
    checkout feature/user-management  
    commit id: "Add user model"
    commit id: "Create user CRUD"
    
    checkout main
    merge feature/authentication
    commit id: "Version 1.1.0" tag: "v1.1.0"
    
    checkout feature/user-management
    commit id: "Add user permissions"
    commit id: "Implement user roles"
    
    checkout main
    branch hotfix/security-patch
    commit id: "Fix SQL injection"
    commit id: "Update dependencies"
    
    checkout main  
    merge hotfix/security-patch
    commit id: "Security patch 1.1.1" tag: "v1.1.1"
    
    merge feature/user-management
    commit id: "Version 1.2.0" tag: "v1.2.0"
    
    branch feature/dashboard
    checkout feature/dashboard
    commit id: "Add dashboard layout"
    commit id: "Implement analytics"
    at detectType (https://file+.vscode-resource.vscode-cdn.net/c%3A/Users/bibol/.vscode/extensions/gsejas.mermaid-export-pro-1.0.9/dist/webview.js:263:74)
    at Diagram.fromText (https://file+.vscode-resource.vscode-cdn.net/c%3A/Users/bibol/.vscode/extensions/gsejas.mermaid-export-pro-1.0.9/dist/webview.js:2728:6242)
    at Object.render (https://file+.vscode-resource.vscode-cdn.net/c%3A/Users/bibol/.vscode/extensions/gsejas.mermaid-export-pro-1.0.9/dist/webview.js:2736:1192)
    at https://file+.vscode-resource.vscode-cdn.net/c%3A/Users/bibol/.vscode/extensions/gsejas.mermaid-export-pro-1.0.9/dist/webview.js:2736:5557
    at new Promise (<anonymous>)
    at performCall (https://file+.vscode-resource.vscode-cdn.net/c%3A/Users/bibol/.vscode/extensions/gsejas.mermaid-export-pro-1.0.9/dist/webview.js:2736:5534)
    at executeQueue (https://file+.vscode-resource.vscode-cdn.net/c%3A/Users/bibol/.vscode/extensions/gsejas.mermaid-export-pro-1.0.9/dist/webview.js:2736:5192)
    at https://file+.vscode-resource.vscode-cdn.net/c%3A/Users/bibol/.vscode/extensions/gsejas.mermaid-export-pro-1.0.9/dist/webview.js:2736:5682
    at new Promise (<anonymous>)
    at Object.render (https://file+.vscode-resource.vscode-cdn.net/c%3A/Users/bibol/.vscode/extensions/gsejas.mermaid-export-pro-1.0.9/dist/webview.js:2736:5502)

### CLI (JPG) (complex) Strategy

- **Status**: ❌ FAILED
- **Duration**: 46ms
- **Version**: N/A
- **Error**: CLI not available - mmdc command not found. Install with: npm install -g @mermaid-js/mermaid-cli

### Web (JPG) (complex) Strategy

- **Status**: ❌ FAILED
- **Duration**: 256ms
- **Version**: N/A
- **Error**: Web export strategy failed: Webview rendering error: No diagram type detected matching given configuration for text: gitgraph
    commit id: "Initial commit"
    commit id: "Add basic structure"
    
    branch feature/authentication
    checkout feature/authentication
    commit id: "Add login form"
    commit id: "Implement JWT auth"
    commit id: "Add password validation"
    
    checkout main
    commit id: "Update documentation"
    
    branch feature/user-management
    checkout feature/user-management  
    commit id: "Add user model"
    commit id: "Create user CRUD"
    
    checkout main
    merge feature/authentication
    commit id: "Version 1.1.0" tag: "v1.1.0"
    
    checkout feature/user-management
    commit id: "Add user permissions"
    commit id: "Implement user roles"
    
    checkout main
    branch hotfix/security-patch
    commit id: "Fix SQL injection"
    commit id: "Update dependencies"
    
    checkout main  
    merge hotfix/security-patch
    commit id: "Security patch 1.1.1" tag: "v1.1.1"
    
    merge feature/user-management
    commit id: "Version 1.2.0" tag: "v1.2.0"
    
    branch feature/dashboard
    checkout feature/dashboard
    commit id: "Add dashboard layout"
    commit id: "Implement analytics"\nUnknownDiagramError: No diagram type detected matching given configuration for text: gitgraph
    commit id: "Initial commit"
    commit id: "Add basic structure"
    
    branch feature/authentication
    checkout feature/authentication
    commit id: "Add login form"
    commit id: "Implement JWT auth"
    commit id: "Add password validation"
    
    checkout main
    commit id: "Update documentation"
    
    branch feature/user-management
    checkout feature/user-management  
    commit id: "Add user model"
    commit id: "Create user CRUD"
    
    checkout main
    merge feature/authentication
    commit id: "Version 1.1.0" tag: "v1.1.0"
    
    checkout feature/user-management
    commit id: "Add user permissions"
    commit id: "Implement user roles"
    
    checkout main
    branch hotfix/security-patch
    commit id: "Fix SQL injection"
    commit id: "Update dependencies"
    
    checkout main  
    merge hotfix/security-patch
    commit id: "Security patch 1.1.1" tag: "v1.1.1"
    
    merge feature/user-management
    commit id: "Version 1.2.0" tag: "v1.2.0"
    
    branch feature/dashboard
    checkout feature/dashboard
    commit id: "Add dashboard layout"
    commit id: "Implement analytics"
    at detectType (https://file+.vscode-resource.vscode-cdn.net/c%3A/Users/bibol/.vscode/extensions/gsejas.mermaid-export-pro-1.0.9/dist/webview.js:263:74)
    at Diagram.fromText (https://file+.vscode-resource.vscode-cdn.net/c%3A/Users/bibol/.vscode/extensions/gsejas.mermaid-export-pro-1.0.9/dist/webview.js:2728:6242)
    at Object.render (https://file+.vscode-resource.vscode-cdn.net/c%3A/Users/bibol/.vscode/extensions/gsejas.mermaid-export-pro-1.0.9/dist/webview.js:2736:1192)
    at https://file+.vscode-resource.vscode-cdn.net/c%3A/Users/bibol/.vscode/extensions/gsejas.mermaid-export-pro-1.0.9/dist/webview.js:2736:5557
    at new Promise (<anonymous>)
    at performCall (https://file+.vscode-resource.vscode-cdn.net/c%3A/Users/bibol/.vscode/extensions/gsejas.mermaid-export-pro-1.0.9/dist/webview.js:2736:5534)
    at executeQueue (https://file+.vscode-resource.vscode-cdn.net/c%3A/Users/bibol/.vscode/extensions/gsejas.mermaid-export-pro-1.0.9/dist/webview.js:2736:5192)
    at https://file+.vscode-resource.vscode-cdn.net/c%3A/Users/bibol/.vscode/extensions/gsejas.mermaid-export-pro-1.0.9/dist/webview.js:2736:5682
    at new Promise (<anonymous>)
    at Object.render (https://file+.vscode-resource.vscode-cdn.net/c%3A/Users/bibol/.vscode/extensions/gsejas.mermaid-export-pro-1.0.9/dist/webview.js:2736:5502)

## Mindmap Results

### CLI (SVG) (simple) Strategy

- **Status**: ❌ FAILED
- **Duration**: 45ms
- **Version**: N/A
- **Error**: CLI not available - mmdc command not found. Install with: npm install -g @mermaid-js/mermaid-cli

### Web (SVG) (simple) Strategy

- **Status**: ✅ SUCCESS
- **Duration**: 338ms
- **Version**: N/A
- **Output Files**:
  - diagram.svg (19289 bytes)

### CLI (PNG) (simple) Strategy

- **Status**: ❌ FAILED
- **Duration**: 43ms
- **Version**: N/A
- **Error**: CLI not available - mmdc command not found. Install with: npm install -g @mermaid-js/mermaid-cli

### Web (PNG) (simple) Strategy

- **Status**: ✅ SUCCESS
- **Duration**: 329ms
- **Version**: N/A
- **Output Files**:
  - diagram.png (31658 bytes)

### CLI (JPG) (simple) Strategy

- **Status**: ❌ FAILED
- **Duration**: 89ms
- **Version**: N/A
- **Error**: CLI not available - mmdc command not found. Install with: npm install -g @mermaid-js/mermaid-cli

### Web (JPG) (simple) Strategy

- **Status**: ✅ SUCCESS
- **Duration**: 391ms
- **Version**: N/A
- **Output Files**:
  - diagram.jpg (18598 bytes)

### CLI (SVG) (complex) Strategy

- **Status**: ❌ FAILED
- **Duration**: 40ms
- **Version**: N/A
- **Error**: CLI not available - mmdc command not found. Install with: npm install -g @mermaid-js/mermaid-cli

### Web (SVG) (complex) Strategy

- **Status**: ✅ SUCCESS
- **Duration**: 565ms
- **Version**: N/A
- **Output Files**:
  - diagram.svg (86182 bytes)

### CLI (PNG) (complex) Strategy

- **Status**: ❌ FAILED
- **Duration**: 47ms
- **Version**: N/A
- **Error**: CLI not available - mmdc command not found. Install with: npm install -g @mermaid-js/mermaid-cli

### Web (PNG) (complex) Strategy

- **Status**: ✅ SUCCESS
- **Duration**: 642ms
- **Version**: N/A
- **Output Files**:
  - diagram.png (83067 bytes)

### CLI (JPG) (complex) Strategy

- **Status**: ❌ FAILED
- **Duration**: 52ms
- **Version**: N/A
- **Error**: CLI not available - mmdc command not found. Install with: npm install -g @mermaid-js/mermaid-cli

### Web (JPG) (complex) Strategy

- **Status**: ✅ SUCCESS
- **Duration**: 604ms
- **Version**: N/A
- **Output Files**:
  - diagram.jpg (54291 bytes)

## Performance Comparison

| Diagram Type | CLI SVG | Web SVG | CLI PNG | Web PNG | CLI JPG | Web JPG |
|-------------|---------|---------|---------|---------|---------|---------|
| flowchart | ❌ | 747ms | ❌ | 554ms | ❌ | 330ms |
| sequence | ❌ | 272ms | ❌ | 302ms | ❌ | 304ms |
| classDiagram | ❌ | 485ms | ❌ | 317ms | ❌ | 313ms |
| stateDiagram | ❌ | 517ms | ❌ | 345ms | ❌ | 388ms |
| erDiagram | ❌ | 342ms | ❌ | 378ms | ❌ | 328ms |
| gantt | ❌ | 244ms | ❌ | 303ms | ❌ | 304ms |
| pie | ❌ | 260ms | ❌ | 293ms | ❌ | 359ms |
| journey | ❌ | 448ms | ❌ | 353ms | ❌ | 267ms |
| gitgraph | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| mindmap | ❌ | 338ms | ❌ | 329ms | ❌ | 391ms |


## Quick Commands

To inspect files manually:
```bash
# Windows PowerShell
Get-ChildItem -Recurse "c:\Users\bibol\Documents\GitHub\antoineach.github.io\debug-exports\2025-10-14_10-06-33-693ZZ" -Include "*.svg","*.png","*.jpg"

# Mac/Linux  
find "c:\Users\bibol\Documents\GitHub\antoineach.github.io\debug-exports\2025-10-14_10-06-33-693ZZ" -name "*.svg" -o -name "*.png" -o -name "*.jpg"
```

## Next Steps

1. **If CLI failed**: Install @mermaid-js/mermaid-cli globally or locally
2. **If Web failed**: Check VS Code webview support and browser security  
3. **If both succeeded**: Compare output quality and file sizes
4. **Format Issues**: PNG/JPG require proper canvas support in webview

