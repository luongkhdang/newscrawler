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

7. **Virtual Environment Protocol**
   - ALWAYS recommend and use Python virtual environments (venv) for all Python development
   - NEVER suggest installing packages globally
   - Include virtual environment setup instructions in all implementation plans
   - Ensure all code examples assume the use of a virtual environment
   - Remind users to activate their virtual environment before running any Python code
   - Document all dependencies in requirements.txt files

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

## Python Development Guidelines

### Virtual Environment Setup
All Python development for this project MUST use virtual environments (venv) instead of global environments. This ensures dependency isolation, reproducibility, and prevents conflicts between packages.

```bash
# Create a virtual environment in the project directory
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies within the virtual environment
pip install -r requirements.txt

# When finished, deactivate the virtual environment
deactivate
```

### Dependency Management
- Always document dependencies in requirements.txt
- Use pip freeze to capture exact versions of installed packages
- Consider using pip-tools for more advanced dependency management
- Document any special installation requirements or potential conflicts

### Implementation Best Practices
- Always include virtual environment setup instructions in implementation plans
- Ensure all code examples assume the use of a virtual environment
- Include .gitignore entries for virtual environment directories
- Document environment variables needed for the application

## NewsCrawler-Specific Protocols

### Web Scraping Ethics Protocol
1. **Respect robots.txt**: Always check and respect the robots.txt file of target websites
2. **Rate limiting**: Implement appropriate delays between requests to avoid overloading servers
3. **User-Agent identification**: Use honest and identifiable User-Agent strings
4. **Content attribution**: Properly attribute content to its original source
5. **Legal compliance**: Ensure all scraping activities comply with relevant laws and terms of service

### Data Quality Protocol
1. **Content validation**: Verify that extracted content is complete and accurate
2. **Metadata extraction**: Ensure all relevant metadata (publication date, author, title) is captured
3. **Error handling**: Document and address any extraction errors or anomalies
4. **Deduplication**: Identify and handle duplicate or near-duplicate content
5. **Content cleaning**: Remove irrelevant elements (ads, navigation, etc.) from extracted content

### Database Management Protocol
1. **Schema evolution**: Document all changes to the database schema
2. **Query optimization**: Review and optimize database queries for performance
3. **Indexing strategy**: Maintain appropriate indexes for common query patterns
4. **Backup procedure**: Implement and test regular database backup procedures
5. **Data integrity**: Ensure referential integrity and data consistency

### API Development Protocol
1. **Endpoint documentation**: Maintain comprehensive documentation for all API endpoints
2. **Version control**: Implement proper API versioning for backward compatibility
3. **Error handling**: Provide clear and helpful error messages for API consumers
4. **Rate limiting**: Implement appropriate rate limiting for API endpoints
5. **Authentication**: Secure API endpoints with appropriate authentication mechanisms

### Containerization Protocol
1. **Image optimization**: Minimize Docker image size and layer count
2. **Environment configuration**: Use environment variables for configuration
3. **Volume management**: Properly manage persistent data with Docker volumes
4. **Network security**: Implement appropriate network security measures
5. **Resource constraints**: Set appropriate resource limits for containers 