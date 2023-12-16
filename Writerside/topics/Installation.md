# Installation

<note>
Currently <code>mmcore</code> is compatible with Python 3.9 and newer
</note>
Before we begin, ensure that the <code>mmcore</code> library is installed.


<procedure title="Docker">
<step>
Preferred method of installation is docker.
<code-block lang="bash">
docker pull ghcr.io/contextmachine/mmcore:main
</code-block>
</step>

<step>
Now you can use dev-container with `mmcore` during development. Or build images of your own applications and 
services for
production.
</step>
</procedure>

<procedure title="Poetry">
<step>
The second fine way of installing it, assuming you're using poetry.

Add mmcore in project dependencies
<code-block lang="bash">
poetry add git+https://github.com/contextmachine/mmcore.git

</code-block>
</step>
<step>
And then just install using poetry.
<code-block lang="bash">
poetry install
</code-block>
</step>
<step>
Now you can import it.
<code-block lang="python">
import mmcore
print(mmcore.__version__())
</code-block>
</step>
</procedure>


<procedure title="Pip">
<step>
Also, you can install it using pip.
<code-block lang="bash">
python3 -m pip install git+https://github.com/contextmachine/mmcore.git
</code-block>

</step>
<step>
Now you can import it.
<code-block lang="python">
import mmcore
print(mmcore.__version__())
</code-block>
</step>

</procedure>
