using Printf

read_opaque_subsystem(file::Union{MatlabHDF5File,HDF5.File}) = process_mcos(read_subsystem_mcos(file))

read_subsystem_mcos(file::MatlabHDF5File) = read_subsystem_mcos(file.plain)

function read_subsystem_mcos(file::HDF5.File)
    subsystem_group = file["#subsystem#"]::HDF5.Group
    dset = subsystem_group["MCOS"]::HDF5.Dataset
    @assert read_attribute(dset, name_type_attr_matlab) == "FileWrapper__"
    # FileWrapper__ class
    refs = read(dset, Reference)::Array{HDF5.Reference,2}
    mcos = Vector{Any}(undef, length(refs))
    for i = 1:length(refs)
        ds = file[refs[i]]
        try
            mcos[i] = m_read(ds)
        finally
            close(ds)
        end
    end
    close(dset)
    close(subsystem_group)
    return mcos
end

function process_mcos(mcos::Vector{Any})
    fdata = IOBuffer(vec(mcos[1]::Array{UInt8,2}))
    segments, strs = parse_header(fdata)
    classes = parse_class_info(fdata,strs,segments[1],segments[2])
    seg2_props = parse_properties(fdata,strs,mcos,segments[2],segments[3])
    obj_info = parse_object_info(fdata,segments[3],segments[4])
    seg4_props = parse_properties(fdata,strs,mcos,segments[4],segments[5])
    parse_segment5(fdata,segments[5],segments[6])
    objs = Vector{Dict{String,Any}}(undef,length(obj_info))
    for (i,info) in enumerate(obj_info)
        # Get the property from either segment 2 or segment 4
        props = info[2] > 0 ? seg2_props[info[2]] : seg4_props[info[3]]
        # And merge it with the matfile defaults for this class
        objs[i] = merge(mcos[end][info[1]+1]::Dict{String,Any},props)
    end
    return map(x -> classes[x[1]][2],obj_info), objs
end

function parse_header(f::IO)
    seek(f,0)
    id = read(f,UInt32) # First element is a version number? Always 2?
    id == 2 || id == 3 || error("unknown first field (version/id?): ", id)
    # Second element is the number of strings
    n_strs = read(f,UInt32)
    # Followed by up to 6 segment offsets (the last two segments seem to be unused)
    offsets = read!(f,Vector{UInt32}(undef,6))
    # And two reserved fields
    read(f,UInt32) == read(f,UInt32) == 0 || error("reserved header fields nonzero")
    # And now we're at the string data segment
    @assert position(f) == 0x28
    strs = Vector{String}(undef,n_strs)
    for i = 1:n_strs
        # simply delimited by nulls
        strs[i] = readuntil(f, '\0')
    end
    return offsets, strs
end

function parse_class_info(f::IO, strs::Vector{String}, seg_start::UInt32, seg_end::UInt32)
    seek(f,seg_start)
    # The first four int32s unknown. Always 0? Or is this simply an empty slot for another class?
    read(f,UInt32) == read(f,UInt32) == read(f,UInt32) == read(f,UInt32) == 0 ||
        error("unknown header for class information")
    classes = Vector{Tuple{String,String}}(undef,0)
    while position(f) < seg_end
        package_idx = read(f,UInt32)
        package = package_idx > 0 ? strs[package_idx] : ""
        name_idx = read(f,UInt32)
        name = name_idx > 0 ? strs[name_idx] : ""
        read(f,UInt32) == read(f,UInt32) == 0 ||
            error("discovered a nonzero class property for ",name)
        push!(classes,(package, name))
    end
    return classes
end

function parse_properties(f::IO, names::Vector{String}, heap::Vector{Any}, seg_start::UInt32, seg_end::UInt32)
    seek(f,seg_start)
    props = Vector{Dict{String,Any}}(undef,0)
    position(f) >= seg_end && return props
    read(f,Int32) == read(f,Int32) == 0 || error("unknown header for properties segment")
    # sizehint: 8 int32s would be 2 props per object; this is overly generous
    sizehint!(props,ceil(Int,(seg_end-position(f))/(8*4)))
    while position(f) < seg_end
        # For each class, there is first a Int32 describing the number of properties
        nprops = read(f,Int32)
        d = sizehint!(Dict{String,Any}(),nprops)
        for i = 1:nprops
            # For each property, there is an index into our strings
            name_idx = read(f,Int32)
            # A flag describing how the heap_idx is to be interpreted
            flag = read(f,Int32)
            # And a value; often an index into some data structure
            heap_idx = read(f,Int32)
            if flag == 0
                # This means that the property is stored in the names array
                d[names[name_idx]] = names[heap_idx]
            elseif flag == 1
                # The property is stored in the MCOS FileWrapper__ heap
                d[names[name_idx]] = heap[heap_idx+3] # But... the index is off by 3!? Crazy.
            elseif flag == 2
                # The property is a boolean, and the heap_idx itself is the value
                @assert 0 <= heap_idx <= 1 "boolean flag has a value other than 0 or 1"
                d[names[name_idx]] = bool(heap_idx)
            else
                error("unknown flag ",flag," for property ",names[name_idx]," with heap index ",heap_idx)
            end
        end
        push!(props,d)
        # Jump to the next 8-byte aligned offset
        if position(f) % 8 != 0
            seek(f,ceil(Int,position(f)/8)*8)
        end
    end
    return props
end

function parse_object_info(f::IO, seg_start::UInt32, seg_end::UInt32)
    seek(f,seg_start)
    # The first six int32s unknown. Always 0? Or perhaps reserved space for an extra elt?
    read(f,UInt32) == read(f,UInt32) == read(f,UInt32) ==
    read(f,UInt32) == read(f,UInt32) == read(f,UInt32) == 0 ||
        error("unknown header for object information")
    object_info = Vector{Tuple{Int,Int,Int,Int}}(undef,0)
    while position(f) < seg_end
        class_idx = read(f,Int32)
        unknown1 = read(f,Int32)
        unknown2 = read(f,Int32)
        segment1_idx = read(f,Int32) # The index into segment 2
        segment2_idx = read(f,Int32) # The index into segment 4
        obj_id = read(f,Int32)
        @assert unknown1 == unknown2 == 0 "discovered a nonzero object property"
        push!(object_info, (class_idx,segment1_idx,segment2_idx,obj_id))
    end
    return object_info
end

function parse_segment5(f::IO, seg_start::UInt32, seg_end::UInt32)
    seek(f,seg_start)
    seg5 = read!(f,Vector{UInt8}(undef,seg_end-position(f)))
    if any(seg5 .!= 0)
        xxd(seg5)
    end
    @assert seg_end == position(f) && eof(f) "there's more data to be had!"
end

cleanascii!(A::Array{UInt8,N}) where {N} = (A[(A .< 0x20) .| (A .> 0x7e)] .= UInt8('.'); A)

function xxd(x, start=1, stop=length(x))
    for i = div(start-1,8)*8+1:8:stop
        row = i:i+7
        # hexadecimal
        @printf("%04x: ",i-1)
        for r=row
            start <= r <= stop ? @printf("%02x",x[r]) : print("  ")
            r % 4 == 0 && print(" ")
        end
        # ASCII
        print("   ",String(cleanascii!(x[i:min(i+7,end)]))," ")
        # Int32
        for j=i:4:i+7
            start <= j && j+3 <= stop ? @printf("% 12d ",reinterpret(Int32,x[j:j+3])[1]) : print(" "^12)
        end
        println()
    end
end

####

read_opaque_obj(file::MatlabHDF5File, obj_num::Integer) = read_opaque_obj(file.plain, obj_num)

function read_opaque_obj(file::HDF5.File, obj_num::Integer)
    subsystem_group = file["#subsystem#"]::HDF5.Group
    mcos_dset = subsystem_group["MCOS"]::HDF5.Dataset
    mcos_refs = read(mcos_dset, Reference)::Array{HDF5.Reference,2}
    close(mcos_dset)
    close(subsystem_group)
    #
    ds = file[mcos_refs[1]]
    fdata = IOBuffer(vec(m_read(ds)::Array{UInt8,2}))
    close(ds)
    # get property info
    segments, strs = parse_header(fdata)
    class_idx, seg1_idx, seg2_idx, obj_id = parse_object_info(fdata,Int(obj_num),segments[3],segments[4])
    if seg1_idx > 0 # get data from 1st prop segment
        props = parse_properties(fdata,seg1_idx,segments[2],segments[3])
    else # get data from 2nd prop segment
        props = parse_properties(fdata,seg2_idx,segments[4],segments[5])
    end
    # extract data
    data = Dict{String,Any}()
    for (name_idx, flag, heap_idx) in props
        if flag == 0 # This means that the property is stored in the strs array
            data[strs[name_idx]] = strs[heap_idx]
        elseif flag == 1 # The property is stored in the MCOS FileWrapper__ heap
            ds = file[mcos_refs[heap_idx+3]] # But... the index is off by 3!? Crazy.
            data[strs[name_idx]] = m_read(ds)
            close(ds)
        elseif flag == 2 # The property is a boolean, and the heap_idx itself is the value
            @assert 0 <= heap_idx <= 1 "boolean flag has a value other than 0 or 1"
            data[strs[name_idx]] = bool(heap_idx)
        else
            error("unknown flag ",flag," for property ",strs[name_idx]," with heap index ",heap_idx)
        end
    end
    # merge with common properties for this class
    ds = file[mcos_refs[end]]
    data = merge(m_read(ds)[class_idx+1]::Dict{String,Any},data)
    close(ds)
    # get class info
    package_idx, name_idx = parse_class_info(fdata,class_idx,segments[1],segments[2])
    return strs[name_idx], data
end

function parse_object_info(f::IO, obj_num::Int, seg_start::UInt32, seg_end::UInt32)
    # seek to right position
    pos = seg_start + obj_num*6*4
    @assert pos < seg_end "invalid obj_num"
    seek(f,pos)
    # get data (6 Int32)
    class_idx = read(f,Int32)
    unknown1 = read(f,Int32)
    unknown2 = read(f,Int32)
    seg1_idx = read(f,Int32) # The index into segment 2
    seg2_idx = read(f,Int32) # The index into segment 4
    obj_id = read(f,Int32)
    @assert unknown1 == unknown2 == 0 "discovered a nonzero object property"
    return class_idx, seg1_idx, seg2_idx, obj_id
end

function parse_class_info(f::IO, idx::Int32, seg_start::UInt32, seg_end::UInt32)
    # seek to right position
    pos = seg_start + idx*4*4
    @assert pos < seg_end "invalid class_idx"
    seek(f,pos)
    # get data (4 Int32)
    package_idx = read(f,UInt32)
    name_idx = read(f,UInt32)
    @assert read(f,UInt32) == read(f,UInt32) == 0 "discovered a nonzero class property"
    return package_idx, name_idx
end

function parse_properties(f::IO, idx::Int32, seg_start::UInt32, seg_end::UInt32)
    # seek to start position
    seek(f,seg_start)
    read(f,Int32) == read(f,Int32) == 0 || error("unknown header for properties segment")
    # we neek to look at each set of props before the one we want to seek to the right position
    for p = 1:(idx-1)
        nprops = read(f,Int32)
        seek(f, ceil(Int,(position(f)+4*3*nprops)/8)*8)
    end
    @assert position(f) < seg_end "invalid property segment idx"
    # now we are at the right position
    nprops = read(f,Int32)
    prop = Vector{Tuple{Int32,Int32,Int32}}(undef,nprops)
    for i = 1:nprops
        # For each property, there is an index into our strings
        name_idx = read(f,Int32)
        # A flag describing how the heap_idx is to be interpreted
        flag = read(f,Int32)
        # And a value; often an index into some data structure
        heap_idx = read(f,Int32)
        # store
        prop[i] = (name_idx, flag, heap_idx)
    end
    return prop
end
