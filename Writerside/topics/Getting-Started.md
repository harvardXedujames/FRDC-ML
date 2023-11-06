# Getting Started

<procedure title="Installing the Dev. Environment" id="install">
    <step>Ensure that you have the right version of Python.
        The required Python version can be seen in <code>pyproject.toml</code>
        <code-block lang="ini">
            [tool.poetry.dependencies]
            python = "..."
        </code-block>
    </step>
    <step>Start by cloning our repository.
        <code-block lang="shell">
          git clone https://github.com/Forest-Recovery-Digital-Companion/FRDC-ML.git
        </code-block>
    </step>
    <step>Then, create a Python Virtual Env <code>pyvenv</code>
        <tabs>
        <tab title="Windows">
          <code-block lang="shell">python -m venv venv/</code-block>
        </tab>
        <tab title="Linux">
          <code-block lang="shell">python3 -m venv venv/</code-block>
        </tab>
        </tabs> 
    </step>
    <step>
        <a href="https://python-poetry.org/docs/">Install Poetry</a>
        Then check if it's installed with
        <code-block lang="shell">poetry --version</code-block>
        <warning>
        If <code>poetry</code> is not found, it's likely not in the user PATH.
        </warning>
    </step>
    <step>Activate the virtual environment
        <tabs>
        <tab title="Windows">
        <code-block lang="shell">
            cd venv/Scripts
            activate
            cd ../..
        </code-block>
        </tab>
        <tab title="Linux">
        <code-block lang="shell">
            source venv/bin/activate
        </code-block>
        </tab>
        </tabs> 
    </step>
    <step>Install the dependencies. You should be in the same directory as
        <code>pyproject.toml</code>
        <code-block lang="shell">
            poetry install --with dev
        </code-block>
    </step>
    <step>Install Pre-Commit Hooks
        <code-block lang="shell">
            pre-commit install
        </code-block>
    </step>
</procedure>

<procedure title="Setting Up Google Cloud" id="gcloud">
    <step>
        We use Google Cloud to store our datasets. To set up Google Cloud,
        <a href="https://cloud.google.com/sdk/docs/install">
          install the Google Cloud CLI
        </a>
    </step>
    <step>
        Then,
        <a href="https://cloud.google.com/sdk/docs/initializing">
          authenticate your account
        </a>.
        <code-block lang="shell">gcloud auth login</code-block>
    </step>
    <step>
        Finally, 
        <a href="https://cloud.google.com/docs/authentication/provide-credentials-adc">
          set up Application Default Credentials (ADC)
        </a>.
        <code-block lang="shell">gcloud auth application-default login</code-block>
    </step>
    <step>
        To make sure everything is working, <a anchor="tests">run the tests</a>.
    </step>
</procedure>

<procedure title="Pre-commit Hooks" collapsible="true">
    <note>This is optional but recommended.
    Pre-commit hooks are a way to ensure that your code is formatted correctly.
    This is done by running a series of checks before you commit your code.
    </note>
    <step>
        <code-block lang="shell">
            pre-commit install
        </code-block>
    </step>
</procedure>

<procedure title="Running the Tests" collapsible="true" id="tests">
    <step>
        Run the tests to make sure everything is working
        <code-block lang="shell">
            pytest
        </code-block>
    </step>
<step>
    In case of errors:
    <deflist>
        <def title="google.auth.exceptions.DefaultCredentialsError">
            If you get this error, it means that you haven't authenticated your
            Google Cloud account.
            See <a anchor="gcloud">Setting Up Google Cloud</a>
        </def>
        <def title="ModuleNotFoundError" collapsible="true">
            If you get this error, it means that you haven't installed the
            dependencies.
            See <a anchor="install">Installing the Dev. Environment</a>
        </def>
    </deflist>
</step>
</procedure>


## Our Repository Structure

Before starting development, take a look at our repository structure. This will
help you understand where to put your code.

```mermaid
graph LR
    FRDC -- " Core Dependencies " --> src/frdc/
    FRDC -- " Resources " --> rsc/
    FRDC -- " Pipeline " --> pipeline/
    FRDC -- " Tests " --> tests/
    FRDC -- " Repo Dependencies " --> pyproject.toml,poetry.lock
    src/frdc/ -- " Dataset Loaders " --> ./load/
    src/frdc/ -- " Preprocessing Fn. " --> ./preprocess/
    src/frdc/ -- " Train Deps " --> ./train/
    src/frdc/ -- " Model Architectures " --> ./models/
    rsc/ -- " Datasets ... " --> ./dataset_name/
    pipeline/ -- " Model Training Pipeline " --> ./model_tests/
```

src/frdc/
: Source Code for our package. These are the unit components of our pipeline.

rsc/
: Resources. These are usually cached datasets

pipeline/
: Pipeline code. These are the full ML tests of our pipeline.

tests/
: PyTest tests. These are unit tests & integration tests.

### Unit, Integration, and Pipeline Tests

We have 3 types of tests:

- Unit Tests are usually small, single function tests.
- Integration Tests are larger tests that tests a mock pipeline.
- Pipeline Tests are the true production pipeline tests that will generate a
  model.

### Where Should I contribute?

<deflist>
<def title="Changing a small component">
If you're changing a small component, such as a argument for preprocessing,
a new model architecture, or a new configuration for a dataset, take a look
at the <code>src/frdc/</code> directory.
</def>
<def title="Adding a test">
By adding a new component, you'll need to add a new test. Take a look at the
<code>tests/</code> directory.
</def>
<def title="Changing the pipeline">
If you're a ML Researcher, you'll probably be changing the pipeline. Take a
look at the <code>pipeline/</code> directory.
</def>
<def title="Adding a dependency">
If you're adding a new dependency, use <code>poetry add PACKAGE</code> and
commit the changes to <code>pyproject.toml</code> and <code>poetry.lock</code>.
<note>
    E.g. Adding <code>numpy</code> is the same as 
    <code>poetry add numpy</code>
</note>
</def>
</deflist>