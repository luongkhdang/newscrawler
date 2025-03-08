# AI-Assisted Project Template

## Core Documents

### 1. DOCUMENT.md (Main Project Document)
```markdown
# [Project Name]

## !IMPORTANT – REFERENCE SYSTEM
- **KNOWLEDGE_BASE.md**: Reference point for AI agents when they need to understand concepts, theories, vocabulary, or structures.
- **DOCUMENT.md**: Main project document (this file).
- **AI_WORKSPACE/FINDINGS.md**: Contains research findings and insights discovered by AI agents.
- **AI_WORKSPACE/PROTOCOLS.md**: Contains rules and protocols for all AI agents working on this project.
- **AI_WORKSPACE/QUESTIONS.md**: Repository for user questions directed to AI agents.
- **AI_WORKSPACE/PLANS/**: Directory containing all research and implementation plans created by AI agents.
- **AI_WORKSPACE/SYSTEM_DESIGN.md**: Contains system architecture documentation, diagrams, and performance considerations.
- **AI_WORKSPACE/CODE_QUALITY.md**: Defines engineering standards and code quality guidelines.
- **AI_WORKSPACE/SECURITY_COMPLIANCE.md**: Outlines security practices and compliance requirements.

## !IMPORTANT – AI AGENT RULES
1. When encountering an unfamiliar concept, ALWAYS check KNOWLEDGE_BASE.md first before responding.
2. All research findings must be documented in AI_WORKSPACE/FINDINGS.md using the specified format.
3. Follow all protocols specified in AI_WORKSPACE/PROTOCOLS.md.
4. When uncertain about a user request, refer to AI_WORKSPACE/QUESTIONS.md for clarification.
5. When asked to create a plan, follow the Plan Creation Workflow in AI_WORKSPACE/PROTOCOLS.md and store the plan in AI_WORKSPACE/PLANS/ directory.
6. Optimize content in AI_WORKSPACE/ directory for AI agent consumption (token efficiency/impact ratio) and content outside AI_WORKSPACE/ for human consumption.
7. DO NOT REPEAT INFORMATION. Use cross-references between documents whenever possible to avoid redundancy.
8. Track progress on plans by checking off completed tasks and updating plan status.
9. Use dynamic tagging instead of fixed categories for questions and findings to improve adaptability.
10. Before implementing any code, document the system design in AI_WORKSPACE/SYSTEM_DESIGN.md.
11. Follow code quality guidelines in AI_WORKSPACE/CODE_QUALITY.md for all implementations.
12. Ensure all code adheres to security and compliance standards in AI_WORKSPACE/SECURITY_COMPLIANCE.md.

## Project Overview
[Brief description of the project, its goals, and expected outcomes]

## Project Scope
[Define what is included and excluded from the project]

## Key Components
[List and describe the main components or modules of the project]

## Implementation Strategy
[High-level approach to implementing the project]

## Timeline
[Project timeline with major milestones]

## Success Metrics
[Measurable criteria for determining project success]

## Resources
[Required resources, tools, technologies, and references]
```

### 2. KNOWLEDGE_BASE.md (Technical Reference Document)
```markdown
# Knowledge Base

## Purpose
This document serves as the central reference point for all AI agents working on [Project Name]. When encountering unfamiliar concepts, theories, vocabulary, or structures, AI agents should consult this document first.

## Core Concepts
[List and define the fundamental concepts relevant to the project]

### [Concept 1]
[Detailed explanation with key components]

### [Concept 2]
[Detailed explanation with key components]

## Technical Terminology
[Glossary of technical terms organized by category]

### [Category 1]
- **[Term 1]**: [Definition]
- **[Term 2]**: [Definition]

### [Category 2]
- **[Term 1]**: [Definition]
- **[Term 2]**: [Definition]

## Data Sources
[List and describe relevant data sources]

### [Source Category 1]
- **[Source 1]**: [Description and relevance]
- **[Source 2]**: [Description and relevance]

## Implementation Technologies
[List and describe technologies used in the project]

### [Technology Category 1]
- **[Technology 1]**: [Description and purpose]
- **[Technology 2]**: [Description and purpose]

## Methodological Approaches
[Describe methodologies relevant to the project]

### [Methodology 1]
[Detailed explanation]

### [Methodology 2]
[Detailed explanation]

## Glossary of Terms
[Alphabetical list of all terms with brief definitions]
```

### 3. AI_WORKSPACE/PROTOCOLS.md (AI Agent Guidelines)
```markdown
# AI Protocols

## Core Principles for AI Agents

1. **Document Navigation Protocol**
   - Always reference KNOWLEDGE_BASE.md for technical concepts
   - Update FINDINGS.md with all research findings
   - Direct clarification questions to QUESTIONS.md
   - Follow the structure outlined in DOCUMENT.md

2. **Research Methodology**
   - Prioritize primary sources over secondary interpretations
   - Cross-reference multiple data points before drawing conclusions
   - Maintain comprehensive citation trails for all information
   - Assign confidence scores to all generated insights (Low/Medium/High)

3. **Knowledge Organization**
   - Categorize all findings according to the structure in DOCUMENT.md
   - Tag information with appropriate priority levels
   - Link related concepts across different knowledge domains
   - Prioritize high-impact information for deeper analysis

4. **Response Protocol**
   - Provide direct answers to all user queries
   - Format responses with clear hierarchical structure
   - Include actionable insights whenever possible
   - Indicate confidence level for all provided information

5. **Content Optimization**
   - All content in AI_WORKSPACE/ directory must be optimized for AI agent consumption (token efficiency/impact ratio)
   - All content outside AI_WORKSPACE/ directory must be optimized for human consumption
   - Do not repeat information unnecessarily - use cross-references between documents
   - Maintain consistent terminology across all documents

6. **Cross-Referencing Protocol**
   - Always link to existing information rather than repeating it
   - Use precise references (e.g., "See KNOWLEDGE_BASE.md, Section 3.2")
   - Create new entries only when information doesn't exist elsewhere
   - Regularly audit and update cross-references to maintain coherence

## Plan Creation Workflow

When asked to create a research or implementation plan, AI agents must follow this workflow:

1. **Plan Initialization**
   - Create a new Markdown file in AI_WORKSPACE/PLANS/ with format: `YYYY-MM-DD_PlanTitle.md`
   - Include metadata section with creation date, objective, and estimated completion time
   - Link to relevant sections in KNOWLEDGE_BASE.md and DOCUMENT.md

2. **Plan Structure**
   - Begin with an executive summary (3-5 sentences maximum)
   - Create a hierarchical task breakdown with clear dependencies
   - Assign priority levels (Critical/High/Medium/Low) to each task
   - Include checkboxes for task completion tracking: `- [ ] Task description`

3. **Implementation Tracking**
   - Mark tasks as complete by updating checkboxes: `- [x] Completed task`
   - Document blockers or dependencies in comments below tasks
   - Update plan status at the top of the document (Not Started/In Progress/Completed/Blocked)
   - Cross-reference findings to FINDINGS.md entries

4. **Completion Protocol**
   - When all tasks are complete, update plan status to "Completed"
   - Add a summary of outcomes and insights gained
   - Link to any artifacts or outputs created
   - Document lessons learned for future reference

### Plan Template

```markdown
# [Plan Title]

**Status**: [Not Started/In Progress/Completed/Blocked]
**Created**: YYYY-MM-DD
**Objective**: [1-2 sentence description]
**Estimated Completion**: [Timeframe]
**References**: [Links to relevant documents]

## Executive Summary
[3-5 sentence overview of the plan]

## Tasks

### Phase 1: [Phase Name]
- [ ] **[P0-Critical]** Task description
- [ ] **[P1-High]** Task description
  - Dependency: [Reference to another task or external factor]
- [ ] **[P2-Medium]** Task description

### Phase 2: [Phase Name]
- [ ] **[P1-High]** Task description
- [ ] **[P2-Medium]** Task description
- [ ] **[P3-Low]** Task description

## Outcomes
[To be completed upon plan execution]

## Lessons Learned
[To be completed upon plan execution]
```

## Data Processing Guidelines
[Guidelines for how AI agents should process and analyze data]

## Report Structure
[Standard format for research reports and findings]

## Technical Implementation Notes
[Notes on technical implementation details relevant to AI agents]
```

### 4. AI_WORKSPACE/QUESTIONS.md (Question Repository)
```markdown
# Questions

This document serves as a repository for user questions directed to AI agents. When AI agents encounter unclear requests or need additional information, they should direct users to add their questions here.

## Question Format

Please use the following format when adding questions:

```
## [YYYY-MM-DD] Question Title

**Context**: Brief background information relevant to the question
**Question**: The specific question or request for the AI
**Priority**: High/Medium/Low
**Required by**: Date when answer is needed (if applicable)
**Tags**: [tag1], [tag2], [tag3]
```

## Recent Questions

[Questions will appear here as they are added by users]

## Answered Questions

[Answered questions will be moved here with their responses]

## Dynamic Tagging System

Instead of fixed categories, this document uses a dynamic tagging system. AI agents should:
- Assign relevant tags to each question based on content
- Create new tags as needed for emerging topics
- Cluster similar questions based on tag patterns
- Maintain a list of commonly used tags for reference
```

### 5. AI_WORKSPACE/FINDINGS.md (Research Findings)
```markdown
# Research Findings

This document contains all research findings and insights discovered by AI agents working on [Project Name].

## Finding Format

```
## [YYYY-MM-DD] Finding Title

**Category**: [Relevant category from DOCUMENT.md]
**Confidence**: High/Medium/Low
**Sources**: [List of sources]
**Related Findings**: [Cross-references to related findings]
**Tags**: [tag1], [tag2], [tag3]

### Summary
[Brief summary of the finding (3-5 sentences)]

### Details
[Detailed explanation of the finding]

### Implications
[What this finding means for the project]

### Next Steps
[Recommended actions based on this finding]
```

## Recent Findings

[Findings will appear here as they are discovered]

## Dynamic Tagging System

Instead of fixed categories, this document uses a dynamic tagging system. AI agents should:
- Assign relevant tags to each finding based on content
- Create new tags as needed for emerging topics
- Cluster similar findings based on tag patterns
- Maintain a list of commonly used tags for reference
```

### 6. AI_WORKSPACE/PLANS/SAMPLE_PLAN.md (Sample Plan)
```markdown
# Sample Plan

**Status**: Not Started
**Created**: YYYY-MM-DD
**Objective**: Brief description of the plan's objective and expected outcomes.
**Estimated Completion**: X days/weeks
**References**: 
- [KNOWLEDGE_BASE.md, Section X.Y](#) - Relevant concept reference
- [DOCUMENT.md, Section X.Y](#) - Related project component

## Executive Summary
This is a sample plan template for AI agents to follow when creating research or implementation plans. It demonstrates the proper structure, task organization, priority assignment, and progress tracking methodology.

## Tasks

### Phase 1: Initial Research
- [ ] **[P0-Critical]** Review existing knowledge in KNOWLEDGE_BASE.md related to the plan objective
- [ ] **[P1-High]** Identify knowledge gaps requiring additional research
  - Dependency: Completion of knowledge review
- [ ] **[P2-Medium]** Collect preliminary data from relevant sources

### Phase 2: Analysis & Processing
- [ ] **[P1-High]** Process collected data according to Data Processing Guidelines
- [ ] **[P2-Medium]** Apply relevant analysis frameworks
- [ ] **[P3-Low]** Document preliminary findings in FINDINGS.md

### Phase 3: Implementation
- [ ] **[P0-Critical]** Develop core functionality based on analysis results
- [ ] **[P1-High]** Test implementation against success criteria
- [ ] **[P2-Medium]** Refine approach based on test results

### Phase 4: Documentation & Integration
- [ ] **[P1-High]** Document final methodology and results
- [ ] **[P2-Medium]** Update relevant sections in FINDINGS.md
- [ ] **[P2-Medium]** Create cross-references to new knowledge in existing documents

## Outcomes
[To be completed upon plan execution]

## Lessons Learned
[To be completed upon plan execution]
```

### 7. AI_WORKSPACE/SYSTEM_DESIGN.md (System Architecture Documentation)
```markdown
# System Design Documentation

This document contains system architecture documentation, diagrams, and performance considerations for [Project Name].

## Architecture Overview
[High-level description of the system architecture]

## Architecture Diagrams
- **UML Class Diagrams**: [Diagrams showing class relationships and structures]
- **API Request-Response Flowcharts**: [Diagrams showing API interactions]
- **Database Schema ER Diagrams**: [Entity-relationship diagrams for databases]
- **Sequence Diagrams**: [Diagrams showing interaction sequences]
- **Component Diagrams**: [Diagrams showing system components and their relationships]

## Data Flow
[Description of how data flows through the system]

## Performance Considerations
- **Complexity Analysis**: [Big-O analysis of key algorithms and operations]
- **Bottleneck Identification**: [Identification of potential performance bottlenecks]
- **Optimization Strategies**: [Strategies for optimizing performance]
- **Scalability Considerations**: [How the system will scale with increased load]

## Technology Stack
[Description of the technology stack used in the system]

## Integration Points
[Description of how the system integrates with external systems]

## Deployment Architecture
[Description of how the system will be deployed]
```

### 8. AI_WORKSPACE/CODE_QUALITY.md (Code Quality Guidelines)
```markdown
# Code Quality Guidelines

This document defines engineering standards and code quality guidelines for [Project Name].

## Engineering Standards
- Follow [Specified Coding Style Guide] for all code.
- Ensure all functions and classes have clear, structured docstrings.
- Code must be fully type-annotated.
- Enforce a modular architecture for maintainability.
- Apply the DRY (Don't Repeat Yourself) principle.
- Use meaningful variable and function names.
- Keep functions and methods small and focused on a single responsibility.
- Use virtual envoronment (if applicable) to isolate dependencies and avoid conflicts with global packages.

## AI Code Review Workflow
AI agents must self-audit code for:
- Readability
- Maintainability
- Scalability
- Performance optimizations
- Adherence to project standards

## Documentation Requirements
- All public APIs must be fully documented.
- Complex algorithms must include explanatory comments.
- Architecture decisions must be documented with rationale.
- Code examples should be provided for non-trivial functionality.

## Testing Standards
- All code must have appropriate unit tests.
- Test coverage should be at least 90%.
- Integration tests must be provided for component interactions.
- Edge cases must be explicitly tested.

## Automated Checks
- Run pre-commit hooks to format and lint code.
- Use automated static analysis tools to catch potential issues.
- Implement continuous integration to verify code quality.

## Language-Specific Guidelines
[Language-specific guidelines for each programming language used in the project]
```

### 9. AI_WORKSPACE/SECURITY_COMPLIANCE.md (Security and Compliance Standards)
```markdown
# Security & Compliance Standards

This document outlines security practices and compliance requirements for [Project Name].

## Secure Coding Practices
- Implement input validation and sanitization for all user inputs.
- Follow OWASP Top 10 security guidelines.
- Use parameterized queries for database operations.
- Implement proper error handling without exposing sensitive information.
- Apply the principle of least privilege for all operations.
- Use secure communication protocols (HTTPS, TLS, etc.).
- Implement proper authentication and authorization mechanisms.
- Avoid hardcoded credentials and secrets.

## Role-Based Access Control (RBAC)
- Define clear user roles and permissions.
- Implement access controls at all API endpoints.
- Regularly audit access logs and permissions.
- Follow the principle of least privilege for all user roles.

## Data Protection
- Encrypt sensitive data at rest and in transit.
- Implement proper data backup and recovery procedures.
- Define data retention policies.
- Implement secure data deletion procedures.

## Compliance Requirements
- [List applicable compliance requirements (GDPR, HIPAA, SOC 2, etc.)]
- Document compliance verification procedures.
- Implement regular compliance audits.
- Maintain records of compliance activities.

## Security Testing
- Implement regular security testing (penetration testing, vulnerability scanning, etc.).
- Document security testing procedures and results.
- Address security issues with appropriate priority.

## Incident Response
- Define incident response procedures.
- Document roles and responsibilities for incident response.
- Implement incident reporting mechanisms.
- Conduct post-incident reviews and implement lessons learned.
```

## Directory Structure

```
/
├── DOCUMENT.md                       # Main project document
├── KNOWLEDGE_BASE.md                # Technical reference document
├── AI_WORKSPACE/                    # Directory for AI agent work
│   ├── PROTOCOLS.md                 # AI agent guidelines
│   ├── FINDINGS.md                  # Research findings
│   ├── QUESTIONS.md                 # Question repository
│   ├── PLANS/                       # Directory for plans
│   │   └── SAMPLE_PLAN.md           # Sample plan template
│   ├── SYSTEM_DESIGN.md             # System architecture documentation
│   ├── CODE_QUALITY.md              # Code quality guidelines
│   └── SECURITY_COMPLIANCE.md       # Security and compliance standards
└── [Project-specific files]         # Other project files
```

## Implementation Instructions

1. Copy this template to create a new project structure
2. Customize each document to fit your specific project needs
3. Ensure all AI agents understand the protocols and document structure
4. Maintain consistent cross-referencing between documents
5. Regularly review and update the knowledge base as the project evolves
6. Use dynamic tagging instead of fixed categories for better adaptability
7. Document system design before implementing code
8. Follow code quality guidelines for all implementations
9. Ensure all code adheres to security and compliance standards 