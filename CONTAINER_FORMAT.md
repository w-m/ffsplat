3D Gaussian Splatting is an exciting method to represent radiance fields. The explicit nature of the list of primitives enables simple editing of scenes. The original implementation stores the primitives as 3D points with attributes in .ply files, with float32 values for each attribute. This takes a lot of space, particularly as there is a large number of spherical harmonics attributes (45), which produces a total of 232 Byte per primitive. This has lead to a welath of research into optimizing the storage, and compressing the 3DGS-based scenes. The research methods can be found in the survey on 3DGS compression: 3DGS.zip.

These research methods utilize different tools to reduce the size of the number of Gaussians while trying to achieve the same quality (compaction), or trying to compress the resulting primitives, to store them in as little space as possible (compression).
The research has shown, that combining compaction and compression, we can reduce the total storage size vs vanilla 3DGS ply over 100x. Compaction, in general, is independent of the compression method. We just reduce the number of primitives, yielding usually linear storage space reduction, relative to the number or primitves reduced.
In the compression research, different tools are presented, but often the methods share similarities, or build from the same concepts or building blocks, like vector quantization.
To make 3DGS methods be used in real-world, we need to have formats that are widely supported by both training methods and renderers and interactive editors. In this regard, we have:
- original 3DGS .ply format
- SuperSplat .splat
- Niantic .spz
- gsplat compressed SOG
Standardization efforts are underway to integrate 3DGS into .gltf.
3DGS is the most popular method that has seen a wide adoption through research, and starting to see adoption in industry. Meanwhile, there have sprung up variants that modify the splat representation (like 2DGS), modify the rendering of primitives (EVER, 3DGRT), or propose different non-splat based alternatives for radiance fields, which are still explicit. There we have plenoxels, a method that preceeds 3DGS, and the recent works RadFoam and SVRaster. Finally, there’s the neural and implicit Radiance Field methods, such as Zip-NeRF and Instant NGP.
Research papers often produce single-use formats that try to demonstrate an idea, while not looking for usability and interoperability. With users of the technology missing out on the latest developments.
Our goal here is to provide a format that allows the description of radiance fields pipelines and their storage, to allow a common language and baseline.
We will start with an explicit description of the .ply format used in 3DGS. Next, we’ll describe a compressed 3DGS format based on Self-Organizing Gaussians, which uses PNG images to store the attributes, which makes it simple to be decoded in any software, even web browsers.

It is currently impossible to see which radiance field method will have the widest adoption in a few years. But we hope it helps the community if formalize some of the descriptions to cut through the noise.

To describe a radiance field scene, we need to know what method to use to render the scene, and then we need the data inputs to render it. For NeRF-methods, these would be the network weights. For 3DGS, this is the list of primitives, each with attributes such as means, scales, rotations, opacity and color.

Describing a RF, there are different vantage points, or users. We have the renderer - it requires the data to produce an output image, given some camera parameters. For 3DGS, this is the list of splats with their attributes.

We have the storage view: how is this data being put into files - think .ply for 3DGS. Here, we have a single file storing all the splats, but with different names for the attributes, e.g. „x“, „y“ and „z“ are held seperate, which become „means“ for the renderer. Also the spherical harmonics are split into f_rest_0, f_rest_1 .. f_rest_44 attributes in the ply.

Then we have a decoder view: we need a description that allows us to read the files, and process them into something that can be given to the renderer. This requires description on how to combine xyz into means, and how to scale the opacity values (e.g. with a sigmoid function).

Finally, we have the encoder’s view. For the .ply format these are very straightforward steps. For more involved compression methods such as Self-Organizing Gaussians, we need to do processing to arrive at some storage values. This can involve computationally intensive sorting operations, building vector quantizations, and image compression methods. All these steps have parameters that need to be tweaked per scene, which we need to somehow determine at compression time. The final format does not need to know about these, but may still want to keep them as metadata, to reproduce these results. The encoder also needs to produce the description that is found by the decoder to produce a renderable representation - the scene parameters.

Usually the details of each format are somewhat hidden in code or in a technical report. Our goal is to make these details explicit and visible.

By describing the building blocks of formats such as 3DGS .ply, .spz or gsplat compression, we hope to enable better tooling, interoperability, and faster iteration for novel methods. When the building blocks are in place, it should become much simpler for a novel representation to create a well-compressible and well-readable format.

We attempted to describe gaussian splatting scenes with the gs-container-format, in a file-centric view (github.com/SharkWipf/gaussian-splatting-container-format/). This file-centric view allows the description of what is needed to store on disk, but does not show the steps to decode these files into the scene parameters given to the renderer. This would still be hidden, implementation-specific.

Thus additionally to describing the files stored, we also want to describe explicitly how to decode them into the scene parameters.

This should allow us to build decoders in dynamic languages such as Python, which can decode a whole set of different formats.
But then, we still require dedicated high-performance decoders, to e.g. open 3DGS scenes in editors and viewers.
To enable allowing to build decoders for a fixed set of features, like the very stricly specified .spz which doesn’t allow for lots of wiggle room (TODO wording), we require another description: a schema, that allows to specify which features to expect. We can validate our container metadatadescription against such a schema, and then build high-performance decoders.


To summarize, we want to create a format to describe the parts requried to render a radiance field. How they are stored, what they are, and how they are processed into something given to a particular renderer.
We choose to use YAML as our description format, which is human readable, machine readable, and allows us store store lists and dictionaries in a simple way, while not requiring quotes for either string keys or values.

We will store this metadata next to the stored compressed files. We can then zip up the folder with all the content (either with actual compression or just zip storage if the contained files are already compressed data like PNG), and give it a custom file ending, and MIME type (see .spz github issue #20).

Proposed file ending: .smurfx (Smashingly Marvellous Universal Radiance Field eXchange)

In this format, we should describe general information: a container identifier, the container version, what software was used to create this container (the packer). The packer version can be part of its name. We do not want to specify it separately, as we do not want the decoder to depend on particular packer versions. It should be there for informational purposes of the user, but all information required for decoding MUST come from the other fields.

We then provide a profile that tells us what kind of data and processing steps to expect. This can be something like „3DGS-INRIA.ply“, or „.spz“, or „SOG-web“. The profile gets a version as well, to be able to evolve in the future.

Next, we describe the list of files stored in this container. For .ply format, this is a single descriptor giving us the file name, its type, and how we name the data extracted from it.

- file_name:
  type: ply
  field_prefix: point_cloud.ply@

This would tell a decoder to read this .ply file, take all the containing data attached to vertices, and make them available as fields with a prefix. We could also specify each field explicitly (e.g. for INRIA 3DGS „x“, „y“, „z“, „scale_0“, „scale_1“, …). This would have the upside of being a more explicit desciption, but adds a lot of information to the YAML, which becomes harder to parse for humans. A 3DGS INRIA .ply contains 61 attributes, which is a very long string list. We choose the description of prefix, which will allow the ply file reader to provide each of these with a prefix for further processing, e.g. point_cloud.ply@x for „x“.

TODO: these are only the „vertex“ datat attributes from .ply, for INRIA 3DGS support. Make that explicit somehow in the ply description? We can have other data in ply as well, like face data, metadata?

Another example entry for the files list could be an image, for SOG:
TODO add means.png example

Next, we describe how the data recovered from these files is processed and transformed, into the final scene parameters, which are given to the renderer.


We use notation that is straightforward to translate into operations on PyTorch tensors. This is the common ground in which nearly all radiance field methods are written in currently, thus the community should be immediately familiar with these terms. They are also expressive enough to write dedicated decoders in e.g. WebGL / WebGPU shaders (RFQ: does this create any headaches?).

To support the standard .ply of 3DGS, we need to support the following operations:

Combine: combine several named fields into a new one. We need to support stacking fields, like f_dc_0, f_dc_1, f_dc_2 into f_dc [N x 3]. We need to support concat, e.g. f_dc [ N x 1 x 3] and f_rest [N x 15 x 3] into sh [N x 16 x 3]. The difference is that stack allows to stack them into a new dimension, where every input element has the same shape, while concat allows to stack them in an existing dimension, where the input elements have differing shapes. (NB: stacking could thus be described as adding a new dimension to all the input fields, follow by a concat. But that would require for our f_rest fields, to specify 44x that a new dimension should be added to each of the f_rest_x fields. Having a dedicated stack operation simplifies this).
As the input fields, we either specify a list of names, e.g. [f_dc, f_rest], or a prefix, which requires the decoder to take all the fields whose names start with this prefix, e.g. point_cloud.ply@f_rest_.

Operation order: the list of operations needs to be processed by the decoder from top to bottom to produce the output.
Field order: the fields need to be processed from top to bottom, which ensures that all fields that are required in the next steps already exist.
TODO: how will this be specified for the container format? It would also be possible to build a tree of dependencies to resolve the fields required for each operation and process them dependently. But this puts a large burden on the decoder implementation.

Continuing with the combination operation: for stacking and concat, we need to specify which dimension the operations happens in. And we have a parameter in the combinatoin op that lets us swich between stack or concat.

The next operation is reshape: we reshape the tensor into a target shape. A single dimension may be -1, in which case it’s inferred from the remaining dimensions and the number of elements in the input.

Alternative to combination, we need to specify a from_field operation, that takes another named field as input, for processing, e.g. from_field: point_cloud.ply@opacity.

Finally for .ply, we require a remapping operation, which applies one of a predefined set of functions.
In 3DGS, an exponential activation is used for scaling atributes, and a sigmoid activation is used for the opacity. Thus we create a remapping op, with method either „exp“ or „sigmoid“.

Another entry of the container metadata is a „scene“ field. Here we describe what primitives we expect, and what parameters exist. Not all fields from the decoding process may be required at the rendering stage. Only the ones listed in params need to be given to the renderer, the rest can be discarded. (Optionally an implementation may want to keep them around).

With these building blocks, we can now describe a .ply file:
TODO add exmaple ply YAML

This file is short enough to be well readable, yet contains an explicit description of how to decode the .ply file into scene parameters.



The 3DGS pipeline
The training view: attribute buffers, initialized from SfM, get concatted, their activation methods applied, handed to the renderer. In a backward pass, we apply the inverse of the activation methods.
The decoding view: bitstreams get decoded into buffers, the rest is similar to the forward pass in training.
The encoding view: we take the scene parameters, and run algorithms that produce new buffers or indices. In stages, we perform the same methods as the backward pass (e.g. the inverse activation methods).

Lifecycle of 3DGS attributes.
- initialization from SfM
- setup of a training pipeline: forward pass with activations, rendering, backward pass with inverse activations
- setup of an encoding pipeline: we take the final buffers from training, run the training forward pass. this produces scene parameters. we then run a novel encoding pipeline, which is not identical to the training backward pass, but runs in the same direction. we store the encoded files.
then, for rendering, we load the container, and setup a decoding pipeline. this is similar to a backward pass of the encoding pipeline, but has potentially fewer steps total. this produces scene parameters, which we give to the renderer.
if we want to finetune or continue training on these results, we need to set up yet another pipeline. this could be identical to parts of the decoding pipeline, but is not required to. if we want to set up a different pipeline, we need a backward / encoding pass.
TODO: provide a diagram / flowchart

The goal of the ffsplat tool, is to
- allow decoding of different RF files into a PyTorch representation, which can be rendered
- allow encoding PyTorch representations into dedicated formats, described with .smurfx metadata
- allow conversion between different formats
- provide a core primitves implementation for Gaussians, which can be used by different Python-based implementations, to read, write, and processes Gaussians, before handing them off to the renderer. this is usually custom built in each repository, but could be shared this way.