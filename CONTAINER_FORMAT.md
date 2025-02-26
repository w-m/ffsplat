# 3D Gaussian Splatting Container Format

## Introduction

3D Gaussian Splatting (3DGS) is an exciting method for representing radiance fields. The explicit nature of the list of primitives enables simple scene editing, which implicit or neural methods like Neural Radiance Fields (NeRFs) can't provide. The original implementation by INRIA stores these primitives as 3D points with attributes in `.ply` files, using `float32` values for each attribute. This consumes significant space, particularly due to the large number of spherical harmonics attributes (45), resulting in 232 bytes total per primitive, and Gigabyte-sized files for simple scenes. This has led to a wealth of research into optimizing storage and compressing 3DGS-based scenes. An overview over research directions can be found in this survey paper: [*3DGS.zip: A survey on 3D Gaussian Splatting Compression Methods*](https://w-m.github.io/3dgs-compression-survey/) (Bagdasarian et. al.).

## Compression Methods

These research methods utilize different approaches:

1. **Compaction**: Reducing the number of Gaussians while maintaining similar quality
2. **Compression**: Compressing the resulting primitives to minimize storage requirements

Research has shown that combining compaction and compression can reduce total storage size by over 100x compared to vanilla 3DGS PLY format. Compaction generally yields linear storage space reduction relative to the number of primitives reduced, and is mostly independent of the chosen compression format, so it is not a focus point here.

In compression research, different tools and ideas are presented, but methods often share similarities or build upon the same concepts and building blocks, such as vector quantization.

## Community progress towards a universal 3DGS format

The 3DGS community has been [discussing](https://github.com/mkkellogg/GaussianSplats3D/issues/47) to standardize a universal format, but has not reached a consensus yet.

Standardization efforts are underway to [integrate 3DGS into `.gltf`](https://github.com/KhronosGroup/glTF/issues/2454).

A townhall on 3DGS Standardization was held by the Metavision Standards Forum, with [slides](https://metaverse-standards.org/presentations-videos/#msf_presentations) and [video recording](https://youtu.be/0xdPpKSkO3I) available.

A [Gaussian Splatting Container Format](https://github.com/SharkWipf/gaussian-splatting-container-format) was discussed by the MrNeRF community on Discord around late 2024. This proposal is an evolution of this work. Through trying to build an implementation of the format, we extended the original proposal with the ideas of adding the field pipeline description to the metadata.

Discussions of this proposal are happening in the MrNeRF & Brush Discord channel #gs-container-format. Feedback and active participation is welcome.

[![](https://dcbadge.limes.pink/api/server/https://discord.gg/TbxJST2BbC)](https://discord.gg/TbxJST2BbC)


## Current Formats

To make 3DGS methods usable in real-world applications, we need formats widely supported by training methods, renderers, and interactive editors. Currently available formats include:

- Original 3DGS `.ply` format
- SuperSplat `.splat`
- [Niantic `.spz`](https://scaniverse.com/spz)
- [gsplat compressed SOG](https://docs.gsplat.studio/main/apis/compression.html)



3DGS is the most popular radiance field method and has seen a wide adoption through research, and starting to see adoption in industry. Meanwhile, there have sprung up variants that modify the splat representation (like [2DGS](https://surfsplatting.github.io)), modify the rendering of primitives ([EVER](https://half-potato.gitlab.io/posts/ever/), [3DGRT](https://gaussiantracer.github.io)), or propose different non-splat based alternatives for radiance fields, which are still explicit. There we have [Plenoxels](https://alexyu.net/plenoxels/), a method that preceeds 3DGS, and the recent works [RadFoam](https://radfoam.github.io) and [SVRaster](https://svraster.github.io). Finally, there’s the neural and implicit Radiance Field methods, such as [Zip-NeRF](https://jonbarron.info/zipnerf/) and [Instant NGP](https://github.com/NVlabs/instant-ngp).

Research papers often produce single-use formats that try to demonstrate an idea, while not looking for usability and interoperability. Which leads to users of the technology missing out on the latest developments.
Our goal here is to provide a format that allows the description of radiance fields pipelines and their storage, to allow a common language and baseline.
We will start with an explicit description of the .ply format used in 3DGS. Next, we’ll describe a compressed 3DGS format based on [Self-Organizing Gaussians (SOG)](https://fraunhoferhhi.github.io/Self-Organizing-Gaussians/), which uses standard image codecs to store the attributes, which makes it simple to be decoded in any software, even web browsers.

It is currently impossible to see which radiance field method will have the widest adoption in a few years. Thus many in the industry are hesitant to proceed with standardization (see the 3DGS Townhall by the MSF). But we hope this proposal helps the community if formalize some of the descriptions, bringing clarity to current storage options, and allow novel methods to be compressable with little additional work.

## Vantage points

To describe a radiance field (RF) scene, we need to know what method to use to render the scene, and then we need the data inputs to render it. For NeRF-methods, these would be the network weights. For 3DGS & friends, this is the list of primitives, each with attributes such as means, scales, rotations, opacity and color.

Describing a radiance field, there are different vantage points, or users. We have the **renderer** - it requires the data to produce an output image, given some camera parameters. For 3DGS, this is the list of splatting primitives with their attributes. The renderer takes the primitives and a camera configuration, and produces the splats, then blends them into the final output image.

We have the **storage view**: how is this data being put into files - think .ply for 3DGS. Here, we have a single file storing all the splats, but with different names for the attributes, e.g. „x“, „y“ and „z“ are held seperate, which become „means“ for the renderer. Also the spherical harmonics are split into f_rest_0, f_rest_1 .. f_rest_44 attributes in the ply.

Then we have a **decoder** view: we need a description that allows us to read the files, and process them into something that can be given to the renderer. This requires description on how to combine xyz into means, and how to scale the opacity values (e.g. with a sigmoid function).

Finally, we have the **encoder**’s view. For the .ply format these are very straightforward steps: take the scene parameters, split them into individual one-dimensional lists, store in .ply. For more involved compression methods such as *Self-Organizing Gaussians (SOG)*, we need to do processing to arrive at some storage values. This can involve computationally intensive sorting operations, building vector quantizations, and image compression methods. All these steps have parameters that need to be tweaked per scene, which we need to somehow determine at compression time. The final format does not need to know about these, but may still want to keep them as metadata, to reproduce these results. The encoder also needs to produce the description that is found by the decoder to produce a renderable representation - the scene parameters.

**Usually the details of each format are somewhat hidden in code or in a technical report. Our goal is to make these details explicit and visible. This includes both storage and the processing required to decode the stored files back into scene parameters.**

By describing the building blocks of formats such as 3DGS .ply, .spz or gsplat compression, we hope to enable better tooling, interoperability, and faster iteration for novel methods. When the building blocks are in place, it should become much simpler for a novel representation to create a well-compressible and well-readable format.

**TODO text: this is repeated with the community stuff above**

We attempted to describe gaussian splatting scenes with the gs-container-format, in a file-centric view (github.com/SharkWipf/gaussian-splatting-container-format/). This file-centric view allows the description of what is needed to store on disk, but does not show the steps to decode these files into the scene parameters given to the renderer. This would still be hidden, implementation-specific.

Thus additionally to describing the files stored, we also want to describe explicitly how to decode them into the scene parameters.

This should allow us to build decoders in dynamic languages such as Python, which can decode a whole set of different formats.
But then, we still require dedicated high-performance decoders, to e.g. open 3DGS scenes in editors and viewers.
To enable allowing to build decoders for a fixed set of features, like the very stricly specified .spz which doesn’t allow for lots of wiggle room (TODO wording), we require another description: a schema, that allows to specify which features to expect. We can validate our container metadatadescription against such a schema, and then build high-performance decoders.

## Scene Description Format

**TODO text: this is repeated with the previous section**

To describe radiance field scenes effectively, we need a consistent format that considers different perspectives:

1. **Renderer's View**: Requires data to produce output images given camera parameters. For 3DGS, this means a list of splats with their attributes.

2. **Storage View**: Defines how data is stored in files. For example, 3DGS uses `.ply` files where attributes like position coordinates "x", "y", "z" are stored separately but become "means" for the renderer. Similarly, spherical harmonics are split into `f_rest_0` through `f_rest_44` attributes.

3. **Decoder's View**: Describes how to read files and process them into renderer-compatible format. This includes combining xyz coordinates into means and applying transformations like sigmoid to opacity values.

4. **Encoder's View**: Outlines the steps to process scene data into storage formats. For compression methods like Self-Organizing Gaussians, this involves sorting operations, vector quantization, and image compression with scene-specific parameters.

## Format Goals

Our goals with this format are to:

1. Make format details explicit and visible rather than hidden in code or technical reports
2. Enable better tooling and interoperability between different methods
3. Support faster iteration for novel representation methods
4. Provide clear descriptions for both file storage and decoding steps

Previously, we attempted to describe gaussian splatting scenes with the gs-container-format using a file-centric approach. While this described what to store on disk, it didn't explain how to decode these files into scene parameters for rendering.

We now aim to explicitly describe both the storage format and the decoding process to enable:

- Building flexible decoders in languages like Python that can handle different formats
- Creating dedicated high-performance decoders for specific formats
- Validating container metadata against schemas for consistency

## YAML Metadata Format

We use YAML to describe the components required to render a radiance field, as it is:

- Human-readable
- Machine-readable
- Capable of storing lists and dictionaries simply
- Doesn't require quotes for string keys or values

The metadata file is stored alongside compressed files. The entire package (metadata plus data files) can be zipped (either with compression or just storage if files are already compressed like PNG) and given a custom file extension and MIME type.

**Proposed file extension**: `.smurfx` (Smashingly Marvellous Universal Radiance Field eXchange)

## Container Structure

A container metadata file includes:

1. **General information**:
   - Container identifier
   - Container version
   - Packer (software used to create the container)

2. **Profile**:
   - Format type (e.g., "3DGS-INRIA.ply", ".spz", or "SOG-web")
   - Profile version

3. **Files**:
   - List of files in the container with their types and field naming conventions

For example, a simple 3DGS .ply file entry might look like:

```yaml
- file_path: "/path/to/file.ply"
  type: ply
  field_prefix: point_cloud.ply@
```

As seen in the `3DGS_INRIA_ply_decoding_template.yaml` example, this tells the decoder to read the PLY file and make all vertex attributes available as fields with the specified prefix.

4. **Fields**:
   - Describes how to process and transform data into final scene parameters

5. **Scene**:
   - Specifies the primitive type and required parameters for rendering

## Operations

To support the standard 3DGS `.ply` format, the following operations are needed:

### Operation Types

The operations in our format reflect different types of data transformations that occur during decoding:

- **Remapping**: 1 field → 1 field (transforms values in a field)
- **Concat/Stack**: N fields → 1 field (combines multiple fields into one)
- **Split**: 1 field → N fields (opposite of concat, separates a field)
- **Lookup**: 2 fields → 1 field (uses one field to look up values in another)
- **File decoders**:
  - PLY decoder: 1 file → N fields (extracts multiple attributes)
  - SOG PNG: 1 file → 1 field (converts image data to tensor)

It's difficult to map all operations into a single category since operations can have one or multiple inputs and one or multiple outputs. This is the motivation for having both a files view (how data is stored) and an operations view (how data is processed).

### Combine Operation

This operation merges multiple fields into a single output field. There are two methods:

1. **Stack**: Creates a new dimension and stacks fields along it. For example, stacking `f_dc_0`, `f_dc_1`, and `f_dc_2` into `f_dc [N x 3]`. 

2. **Concat**: Combines fields along an existing dimension. For example, concatenating `f_dc [N x 1 x 3]` and `f_rest [N x 15 x 3]` into `sh [N x 16 x 3]`.

Input fields can be specified in two ways:
- As a list of names: `from_field_list: [f_dc, f_rest]`
- Using a prefix: `from_fields_with_prefix: point_cloud.ply@f_rest_`

From the `3DGS_INRIA_ply_decoding_template.yaml` example:

```yaml
f_dc:
  - combine:
      from_fields_with_prefix: point_cloud.ply@f_dc_
      method: stack
      dim: 1
  - reshape:
      shape: [-1, 1, 3]
```

### Reshape Operation

This operation changes the dimensions of a tensor. A dimension value of -1 means it will be automatically calculated based on the total number of elements.

```yaml
reshape:
  shape: [-1, 1, 3]
```

### From_field Operation

This operation copies data from another named field for further processing:

```yaml
opacities: 
  - from_field: point_cloud.ply@opacity
  - remapping:
      method: sigmoid
```

### Remapping Operation

Applies a mathematical function to transform field values:

- `exp`: Exponential function for scaling attributes
- `sigmoid`: Sigmoid function for opacity values

Example from the template:

```yaml
scales:
  - combine:
      from_fields_with_prefix: point_cloud.ply@scale_
      method: stack
      dim: 1
  - remapping:
      method: exp
```

### Operation Processing Order

Operations are processed sequentially from top to bottom for each field. The field processing itself follows the order defined in the metadata file.

> Note: A simple decoding example in Python may be added later to demonstrate how these operations are implemented in code.

## Scene Representation

The `scene` section in the metadata file defines what primitives we expect and their required parameters:

```yaml
scene:
  primitives: 3DGS-INRIA
  params:
    - means
    - scales
    - opacities
    - quaternions
    - sh
```

These parameters are used to create the appropriate scene representation object that can be passed to a renderer.

## 3DGS Pipeline and Lifecycle

The 3DGS pipeline can be viewed from different perspectives:

### Training Pipeline

1. **Initialization**: Attribute buffers are initialized from Structure from Motion (SfM)
2. **Forward Pass**: Attributes are concatenated, activation functions applied, and passed to renderer
3. **Backward Pass**: Inverse activation methods are applied for optimization

### Encoding Pipeline

1. **Input**: Final trained buffers are processed with the training forward pass
2. **Encoding**: Novel encoding algorithms produce optimized buffers/indices 
3. **Storage**: The encoded files are stored with metadata

### Decoding Pipeline

1. **Input**: Container with encoded files and metadata is loaded
2. **Processing**: Files are decoded into raw field data and transformed according to the field operations
3. **Output**: Scene parameters ready for the renderer

> Note: A simple implementation example showing these pipeline steps may be added later.

## ffsplat Tool Functionality

The ffsplat tool provides a comprehensive solution for working with 3D Gaussian Splatting formats:

1. **Decoding**: Convert various RF formats into PyTorch tensors for rendering
2. **Encoding**: Convert PyTorch representations into efficient storage formats
3. **Format Conversion**: Transform between different 3DGS formats
4. **Core Primitives**: Shared implementation of Gaussian primitives for Python-based applications

This enables a unified approach to handling various radiance field formats while maintaining interoperability between different implementations.

## Complete Format Example

Let's look at a complete example of the 3DGS INRIA PLY format description from `3DGS_INRIA_ply_decoding_template.yaml`:

```yaml
container_identifier: smurfx
container_version: 0.1

# this is a template file for .ply files without any metadata
packer: unknown

profile: 3DGS-INRIA.ply
profile_version: 0.1

meta:

scene:
  primitives: 3DGS-INRIA
  params:
    - means
    - scales
    - opacities
    - quaternions
    - sh

files:
  - file_path: "/placeholder/path/to/your/3dgs_inria_ply_file.ply"
    type: ply
    field_prefix: point_cloud.ply@

fields:
  f_dc:
    - combine:
        from_fields_with_prefix: point_cloud.ply@f_dc_
        method: stack
        dim: 1
    - reshape:
        shape: [-1, 1, 3]
  
  f_rest:
    - combine:
        from_fields_with_prefix: point_cloud.ply@f_rest_
        method: stack
        dim: 1
    - reshape:
        shape: [-1, 15, 3]

  sh:
    - combine:
        from_field_list: [f_dc, f_rest]  
        method: concat
        dim: 1

  quaternions:
    - combine:
        from_fields_with_prefix: point_cloud.ply@rot_
        method: stack
        dim: 1

  means:
    - combine:
        from_field_list: [point_cloud.ply@x, point_cloud.ply@y, point_cloud.ply@z]
        method: stack
        dim: 1
  
  scales:
    - combine:
        from_fields_with_prefix: point_cloud.ply@scale_
        method: stack
        dim: 1
    - remapping:
        method: exp
  
  opacities: 
    - from_field: point_cloud.ply@opacity
    - remapping:
        method: sigmoid
```

This complete example shows how a standard 3DGS .ply file is decoded into PyTorch tensors ready for rendering. The following operations are performed:

1. The decoder reads the .ply file, extracting all vertex attributes with the prefix `point_cloud.ply@`.
2. The fields are processed according to their operations:
   - Position coordinates (x, y, z) are stacked to form the `means` tensor.
   - Scale values are stacked and then exponentiated.
   - Rotation values are stacked to form quaternions.
   - Color data is processed in two parts:
     - Base colors (`f_dc_0`, `f_dc_1`, `f_dc_2`) are stacked and reshaped.
     - Higher-order spherical harmonics (`f_rest_0` through `f_rest_44`) are stacked and reshaped.
     - These are then concatenated to form the complete `sh` tensor.
   - Opacity values are transformed with a sigmoid function.
3. Finally, the processed fields are used to create a `Gaussians` object ready for rendering.

## Extensions and Future Work

The format can be extended to support other radiance field representations:

### Other Primitives

While the initial focus is on 3DGS, the format could be extended to support:
- 2D Gaussians
- Radiant Foam
- Neural representations (NeRF variants), Instant-NGP
- Hybrid representations

### Schema Validation

A schema system will allow:
- Validation of container metadata
- Feature detection for decoders
- Standardized extensions

## Conclusion

The Radiance Field Container Format provides a flexible, explicit way to describe radiance field scenes and their storage formats. By separating the storage format from the decoding process, we enable:

1. Better interoperability between different implementations
2. Clearer documentation of format specifications
3. Faster development of new compression techniques
4. Simplified sharing of 3D scenes across platforms

The ffsplat tool implements this format, providing a unified approach to handling various radiance field formats while maintaining compatibility with existing workflows.