<?xml version="1.0" ?><graphml xmlns="http://graphml.graphdrawing.org/xmlns" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="http://graphml.graphdrawing.org/xmlns http://graphml.graphdrawing.org/xmlns/1.1/graphml.xsd"><key id="reference" for="node" attr.name="reference" attr.type="string"></key><key id="identifier" for="node" attr.name="identifier" attr.type="string"></key><key id="modifier" for="node" attr.name="modifier" attr.type="string"></key><key id="text" for="node" attr.name="text" attr.type="string"></key><key id="type" for="node" attr.name="type" attr.type="string"></key><key id="parentType" for="node" attr.name="parentType" attr.type="string"></key><key id="type" for="edge" attr.name="type" attr.type="string"></key><graph id="G" edgedefault="directed"><node id="44"><data key="reference">userDefinedMethodName</data><data key="identifier">resetPacketSequence</data><data key="text">resetPacketSequence</data><data key="type">SimpleName</data><data key="parentType">MethodDeclaration</data></node><node id="45"><data key="reference"></data><data key="text">void</data><data key="type">VoidType</data><data key="parentType">MethodDeclaration</data></node><node id="46"><data key="reference"></data><data key="modifier"></data><data key="text">/**
 * Return a PacketReader instance free of decorators.
 *
 * @return
 */
PacketReader undecorateAll();</data><data key="type">MethodDeclaration</data><data key="parentType">ClassOrInterfaceDeclaration</data></node><node id="47"><data key="reference">userDefinedMethodName</data><data key="identifier">undecorateAll</data><data key="text">undecorateAll</data><data key="type">SimpleName</data><data key="parentType">MethodDeclaration</data></node><node id="48"><data key="reference"></data><data key="text">PacketReader</data><data key="type">ClassOrInterfaceType</data><data key="parentType">MethodDeclaration</data></node><node id="49"><data key="reference">nonQualifiedClassName</data><data key="identifier">PacketReader</data><data key="text">PacketReader</data><data key="type">SimpleName</data><data key="parentType">ClassOrInterfaceType</data></node><node id="50"><data key="reference"></data><data key="modifier"></data><data key="text">/**
 * Return the previous PacketReader instance from the decorators chain or the current PacketReader
 * if it is the first entry in a chain.
 *
 * @return
 */
PacketReader undecorate();</data><data key="type">MethodDeclaration</data><data key="parentType">ClassOrInterfaceDeclaration</data></node><node id="51"><data key="reference">userDefinedMethodName</data><data key="identifier">undecorate</data><data key="text">undecorate</data><data key="type">SimpleName</data><data key="parentType">MethodDeclaration</data></node><node id="52"><data key="reference"></data><data key="text">PacketReader</data><data key="type">ClassOrInterfaceType</data><data key="parentType">MethodDeclaration</data></node><node id="53"><data key="reference">nonQualifiedClassName</data><data key="identifier">PacketReader</data><data key="text">PacketReader</data><data key="type">SimpleName</data><data key="parentType">ClassOrInterfaceType</data></node><node id="10"><data key="reference"></data><data key="identifier">IOException</data><data key="text">java.io.IOException</data><data key="type">Name</data><data key="parentType">ImportDeclaration</data></node><node id="11"><data key="reference"></data><data key="identifier">io</data><data key="text">java.io</data><data key="type">Name</data><data key="parentType">Name</data></node><node id="12"><data key="reference"></data><data key="identifier">java</data><data key="text">java</data><data key="type">Name</data><data key="parentType">Name</data></node><node id="13"><data key="reference"></data><data key="text">import java.util.Optional;
</data><data key="type">ImportDeclaration</data><data key="parentType">CompilationUnit</data></node><node id="14"><data key="reference"></data><data key="identifier">Optional</data><data key="text">java.util.Optional</data><data key="type">Name</data><data key="parentType">ImportDeclaration</data></node><node id="15"><data key="reference"></data><data key="identifier">util</data><data key="text">java.util</data><data key="type">Name</data><data key="parentType">Name</data></node><node id="16"><data key="reference"></data><data key="identifier">java</data><data key="text">java</data><data key="type">Name</data><data key="parentType">Name</data></node><node id="17"><data key="reference"></data><data key="modifier">public,interface</data><data key="text">public interface PacketReader {

    /**
     * Read MySQL packet header from input stream.
     *
     * @return
     * @throws IOException
     */
    PacketHeader readHeader() throws IOException;

    /**
     * Read MySQL packet payload from input stream into to the given {@link PacketPayload} instance or into the new one if not present.
     *
     * @param reuse
     *            {@link PacketPayload} to reuse
     * @param packetLength
     *            Expected length of packet
     * @return
     * @throws IOException
     */
    PacketPayload readPayload(Optional&lt;PacketPayload&gt; reuse, int packetLength) throws IOException;

    /**
     * Get last packet sequence number, as it was stored by {@link #readHeader(byte[], boolean)}.
     *
     * @return
     */
    byte getPacketSequence();

    /**
     * Set stored packet sequence number to 0.
     */
    void resetPacketSequence();

    /**
     * Return a PacketReader instance free of decorators.
     *
     * @return
     */
    PacketReader undecorateAll();

    /**
     * Return the previous PacketReader instance from the decorators chain or the current PacketReader
     * if it is the first entry in a chain.
     *
     * @return
     */
    PacketReader undecorate();
}</data><data key="type">ClassOrInterfaceDeclaration</data><data key="parentType">CompilationUnit</data></node><node id="18"><data key="reference"></data><data key="identifier">PacketReader</data><data key="text">PacketReader</data><data key="type">SimpleName</data><data key="parentType">ClassOrInterfaceDeclaration</data></node><node id="19"><data key="reference"></data><data key="modifier"></data><data key="text">/**
 * Read MySQL packet header from input stream.
 *
 * @return
 * @throws IOException
 */
PacketHeader readHeader() throws IOException;</data><data key="type">MethodDeclaration</data><data key="parentType">ClassOrInterfaceDeclaration</data></node><node id="1"><data key="reference"></data><data key="text">/*
  Copyright (c) 2016, Oracle and/or its affiliates. All rights reserved.

  The MySQL Connector/J is licensed under the terms of the GPLv2
  &lt;http://www.gnu.org/licenses/old-licenses/gpl-2.0.html&gt;, like most MySQL Connectors.
  There are special exceptions to the terms and conditions of the GPLv2 as it is applied to
  this software, see the FOSS License Exception
  &lt;http://www.mysql.com/about/legal/licensing/foss-exception.html&gt;.

  This program is free software; you can redistribute it and/or modify it under the terms
  of the GNU General Public License as published by the Free Software Foundation; version 2
  of the License.

  This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
  without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
  See the GNU General Public License for more details.

  You should have received a copy of the GNU General Public License along with this
  program; if not, write to the Free Software Foundation, Inc., 51 Franklin St, Fifth
  Floor, Boston, MA 02110-1301  USA

 */
package com.mysql.cj.api.mysqla.io;

import java.io.IOException;
import java.util.Optional;

public interface PacketReader {

    /**
     * Read MySQL packet header from input stream.
     *
     * @return
     * @throws IOException
     */
    PacketHeader readHeader() throws IOException;

    /**
     * Read MySQL packet payload from input stream into to the given {@link PacketPayload} instance or into the new one if not present.
     *
     * @param reuse
     *            {@link PacketPayload} to reuse
     * @param packetLength
     *            Expected length of packet
     * @return
     * @throws IOException
     */
    PacketPayload readPayload(Optional&lt;PacketPayload&gt; reuse, int packetLength) throws IOException;

    /**
     * Get last packet sequence number, as it was stored by {@link #readHeader(byte[], boolean)}.
     *
     * @return
     */
    byte getPacketSequence();

    /**
     * Set stored packet sequence number to 0.
     */
    void resetPacketSequence();

    /**
     * Return a PacketReader instance free of decorators.
     *
     * @return
     */
    PacketReader undecorateAll();

    /**
     * Return the previous PacketReader instance from the decorators chain or the current PacketReader
     * if it is the first entry in a chain.
     *
     * @return
     */
    PacketReader undecorate();
}
</data><data key="type">CompilationUnit</data></node><node id="2"><data key="reference"></data><data key="text">package com.mysql.cj.api.mysqla.io;

</data><data key="type">PackageDeclaration</data><data key="parentType">CompilationUnit</data></node><node id="3"><data key="reference"></data><data key="identifier">io</data><data key="text">com.mysql.cj.api.mysqla.io</data><data key="type">Name</data><data key="parentType">PackageDeclaration</data></node><node id="4"><data key="reference"></data><data key="identifier">mysqla</data><data key="text">com.mysql.cj.api.mysqla</data><data key="type">Name</data><data key="parentType">Name</data></node><node id="5"><data key="reference"></data><data key="identifier">api</data><data key="text">com.mysql.cj.api</data><data key="type">Name</data><data key="parentType">Name</data></node><node id="6"><data key="reference"></data><data key="identifier">cj</data><data key="text">com.mysql.cj</data><data key="type">Name</data><data key="parentType">Name</data></node><node id="7"><data key="reference"></data><data key="identifier">mysql</data><data key="text">com.mysql</data><data key="type">Name</data><data key="parentType">Name</data></node><node id="8"><data key="reference"></data><data key="identifier">com</data><data key="text">com</data><data key="type">Name</data><data key="parentType">Name</data></node><node id="9"><data key="reference"></data><data key="text">import java.io.IOException;
</data><data key="type">ImportDeclaration</data><data key="parentType">CompilationUnit</data></node><node id="20"><data key="reference">userDefinedMethodName</data><data key="identifier">readHeader</data><data key="text">readHeader</data><data key="type">SimpleName</data><data key="parentType">MethodDeclaration</data></node><node id="21"><data key="reference"></data><data key="text">IOException</data><data key="type">ClassOrInterfaceType</data><data key="parentType">MethodDeclaration</data></node><node id="22"><data key="reference">nonQualifiedClassName</data><data key="identifier">IOException</data><data key="text">IOException</data><data key="type">SimpleName</data><data key="parentType">ClassOrInterfaceType</data></node><node id="23"><data key="reference"></data><data key="text">PacketHeader</data><data key="type">ClassOrInterfaceType</data><data key="parentType">MethodDeclaration</data></node><node id="24"><data key="reference">nonQualifiedClassName</data><data key="identifier">PacketHeader</data><data key="text">PacketHeader</data><data key="type">SimpleName</data><data key="parentType">ClassOrInterfaceType</data></node><node id="25"><data key="reference"></data><data key="modifier"></data><data key="text">/**
 * Read MySQL packet payload from input stream into to the given {@link PacketPayload} instance or into the new one if not present.
 *
 * @param reuse
 *            {@link PacketPayload} to reuse
 * @param packetLength
 *            Expected length of packet
 * @return
 * @throws IOException
 */
PacketPayload readPayload(Optional&lt;PacketPayload&gt; reuse, int packetLength) throws IOException;</data><data key="type">MethodDeclaration</data><data key="parentType">ClassOrInterfaceDeclaration</data></node><node id="26"><data key="reference">userDefinedMethodName</data><data key="identifier">readPayload</data><data key="text">readPayload</data><data key="type">SimpleName</data><data key="parentType">MethodDeclaration</data></node><node id="27"><data key="reference"></data><data key="modifier"></data><data key="text">Optional&lt;PacketPayload&gt; reuse</data><data key="type">Parameter</data><data key="parentType">MethodDeclaration</data></node><node id="28"><data key="reference"></data><data key="text">Optional&lt;PacketPayload&gt;</data><data key="type">ClassOrInterfaceType</data><data key="parentType">Parameter</data></node><node id="29"><data key="reference">nonQualifiedClassName</data><data key="identifier">Optional</data><data key="text">Optional</data><data key="type">SimpleName</data><data key="parentType">ClassOrInterfaceType</data></node><node id="30"><data key="reference"></data><data key="text">PacketPayload</data><data key="type">ClassOrInterfaceType</data><data key="parentType">ClassOrInterfaceType</data></node><node id="31"><data key="reference">nonQualifiedClassName</data><data key="identifier">PacketPayload</data><data key="text">PacketPayload</data><data key="type">SimpleName</data><data key="parentType">ClassOrInterfaceType</data></node><node id="32"><data key="reference">java.util.Optional</data><data key="identifier">reuse</data><data key="text">reuse</data><data key="type">SimpleName</data><data key="parentType">Parameter</data></node><node id="33"><data key="reference"></data><data key="modifier"></data><data key="text">int packetLength</data><data key="type">Parameter</data><data key="parentType">MethodDeclaration</data></node><node id="34"><data key="reference"></data><data key="identifier">int</data><data key="text">int</data><data key="type">PrimitiveType</data><data key="parentType">Parameter</data></node><node id="35"><data key="reference">int</data><data key="identifier">packetLength</data><data key="text">packetLength</data><data key="type">SimpleName</data><data key="parentType">Parameter</data></node><node id="36"><data key="reference"></data><data key="text">IOException</data><data key="type">ClassOrInterfaceType</data><data key="parentType">MethodDeclaration</data></node><node id="37"><data key="reference">nonQualifiedClassName</data><data key="identifier">IOException</data><data key="text">IOException</data><data key="type">SimpleName</data><data key="parentType">ClassOrInterfaceType</data></node><node id="38"><data key="reference"></data><data key="text">PacketPayload</data><data key="type">ClassOrInterfaceType</data><data key="parentType">MethodDeclaration</data></node><node id="39"><data key="reference">nonQualifiedClassName</data><data key="identifier">PacketPayload</data><data key="text">PacketPayload</data><data key="type">SimpleName</data><data key="parentType">ClassOrInterfaceType</data></node><node id="40"><data key="reference"></data><data key="modifier"></data><data key="text">/**
 * Get last packet sequence number, as it was stored by {@link #readHeader(byte[], boolean)}.
 *
 * @return
 */
byte getPacketSequence();</data><data key="type">MethodDeclaration</data><data key="parentType">ClassOrInterfaceDeclaration</data></node><node id="41"><data key="reference">userDefinedMethodName</data><data key="identifier">getPacketSequence</data><data key="text">getPacketSequence</data><data key="type">SimpleName</data><data key="parentType">MethodDeclaration</data></node><node id="42"><data key="reference"></data><data key="identifier">byte</data><data key="text">byte</data><data key="type">PrimitiveType</data><data key="parentType">MethodDeclaration</data></node><node id="43"><data key="reference"></data><data key="modifier"></data><data key="text">/**
 * Set stored packet sequence number to 0.
 */
void resetPacketSequence();</data><data key="type">MethodDeclaration</data><data key="parentType">ClassOrInterfaceDeclaration</data></node><edge id="61" source="44" target="45" label="NEXT_TOKEN"><data key="type">NEXT_TOKEN</data></edge><edge id="67" source="46" target="48" label="AST"><data key="type">AST</data></edge><edge id="65" source="46" target="47" label="AST"><data key="type">AST</data></edge><edge id="64" source="46" target="50" label="NEXT_TOKEN"><data key="type">NEXT_TOKEN</data></edge><edge id="66" source="47" target="48" label="NEXT_TOKEN"><data key="type">NEXT_TOKEN</data></edge><edge id="68" source="48" target="49" label="AST"><data key="type">AST</data></edge><edge id="70" source="50" target="51" label="AST"><data key="type">AST</data></edge><edge id="72" source="50" target="52" label="AST"><data key="type">AST</data></edge><edge id="71" source="51" target="52" label="NEXT_TOKEN"><data key="type">NEXT_TOKEN</data></edge><edge id="73" source="52" target="53" label="AST"><data key="type">AST</data></edge><edge id="11" source="10" target="11" label="AST"><data key="type">AST</data></edge><edge id="12" source="11" target="12" label="AST"><data key="type">AST</data></edge><edge id="15" source="13" target="14" label="AST"><data key="type">AST</data></edge><edge id="14" source="13" target="17" label="NEXT_TOKEN"><data key="type">NEXT_TOKEN</data></edge><edge id="16" source="14" target="15" label="AST"><data key="type">AST</data></edge><edge id="17" source="15" target="16" label="AST"><data key="type">AST</data></edge><edge id="58" source="17" target="43" label="AST"><data key="type">AST</data></edge><edge id="69" source="17" target="50" label="AST"><data key="type">AST</data></edge><edge id="19" source="17" target="18" label="AST"><data key="type">AST</data></edge><edge id="30" source="17" target="25" label="AST"><data key="type">AST</data></edge><edge id="63" source="17" target="46" label="AST"><data key="type">AST</data></edge><edge id="53" source="17" target="40" label="AST"><data key="type">AST</data></edge><edge id="21" source="17" target="19" label="AST"><data key="type">AST</data></edge><edge id="20" source="18" target="19" label="NEXT_TOKEN"><data key="type">NEXT_TOKEN</data></edge><edge id="23" source="19" target="20" label="AST"><data key="type">AST</data></edge><edge id="25" source="19" target="21" label="AST"><data key="type">AST</data></edge><edge id="28" source="19" target="23" label="AST"><data key="type">AST</data></edge><edge id="22" source="19" target="25" label="NEXT_TOKEN"><data key="type">NEXT_TOKEN</data></edge><edge id="0" source="1" target="2" label="AST"><data key="type">AST</data></edge><edge id="13" source="1" target="13" label="AST"><data key="type">AST</data></edge><edge id="18" source="1" target="17" label="AST"><data key="type">AST</data></edge><edge id="8" source="1" target="9" label="AST"><data key="type">AST</data></edge><edge id="2" source="2" target="3" label="AST"><data key="type">AST</data></edge><edge id="1" source="2" target="9" label="NEXT_TOKEN"><data key="type">NEXT_TOKEN</data></edge><edge id="3" source="3" target="4" label="AST"><data key="type">AST</data></edge><edge id="4" source="4" target="5" label="AST"><data key="type">AST</data></edge><edge id="5" source="5" target="6" label="AST"><data key="type">AST</data></edge><edge id="6" source="6" target="7" label="AST"><data key="type">AST</data></edge><edge id="7" source="7" target="8" label="AST"><data key="type">AST</data></edge><edge id="10" source="9" target="10" label="AST"><data key="type">AST</data></edge><edge id="9" source="9" target="13" label="NEXT_TOKEN"><data key="type">NEXT_TOKEN</data></edge><edge id="24" source="20" target="21" label="NEXT_TOKEN"><data key="type">NEXT_TOKEN</data></edge><edge id="27" source="21" target="22" label="AST"><data key="type">AST</data></edge><edge id="26" source="21" target="23" label="NEXT_TOKEN"><data key="type">NEXT_TOKEN</data></edge><edge id="29" source="23" target="24" label="AST"><data key="type">AST</data></edge><edge id="34" source="25" target="27" label="AST"><data key="type">AST</data></edge><edge id="48" source="25" target="36" label="AST"><data key="type">AST</data></edge><edge id="51" source="25" target="38" label="AST"><data key="type">AST</data></edge><edge id="32" source="25" target="26" label="AST"><data key="type">AST</data></edge><edge id="43" source="25" target="33" label="AST"><data key="type">AST</data></edge><edge id="31" source="25" target="40" label="NEXT_TOKEN"><data key="type">NEXT_TOKEN</data></edge><edge id="33" source="26" target="27" label="NEXT_TOKEN"><data key="type">NEXT_TOKEN</data></edge><edge id="36" source="27" target="28" label="AST"><data key="type">AST</data></edge><edge id="42" source="27" target="32" label="AST"><data key="type">AST</data></edge><edge id="35" source="27" target="33" label="NEXT_TOKEN"><data key="type">NEXT_TOKEN</data></edge><edge id="38" source="28" target="29" label="AST"><data key="type">AST</data></edge><edge id="40" source="28" target="30" label="AST"><data key="type">AST</data></edge><edge id="37" source="28" target="32" label="NEXT_TOKEN"><data key="type">NEXT_TOKEN</data></edge><edge id="39" source="29" target="30" label="NEXT_TOKEN"><data key="type">NEXT_TOKEN</data></edge><edge id="41" source="30" target="31" label="AST"><data key="type">AST</data></edge><edge id="45" source="33" target="34" label="AST"><data key="type">AST</data></edge><edge id="47" source="33" target="35" label="AST"><data key="type">AST</data></edge><edge id="44" source="33" target="36" label="NEXT_TOKEN"><data key="type">NEXT_TOKEN</data></edge><edge id="46" source="34" target="35" label="NEXT_TOKEN"><data key="type">NEXT_TOKEN</data></edge><edge id="50" source="36" target="37" label="AST"><data key="type">AST</data></edge><edge id="49" source="36" target="38" label="NEXT_TOKEN"><data key="type">NEXT_TOKEN</data></edge><edge id="52" source="38" target="39" label="AST"><data key="type">AST</data></edge><edge id="55" source="40" target="41" label="AST"><data key="type">AST</data></edge><edge id="57" source="40" target="42" label="AST"><data key="type">AST</data></edge><edge id="54" source="40" target="43" label="NEXT_TOKEN"><data key="type">NEXT_TOKEN</data></edge><edge id="56" source="41" target="42" label="NEXT_TOKEN"><data key="type">NEXT_TOKEN</data></edge><edge id="60" source="43" target="44" label="AST"><data key="type">AST</data></edge><edge id="62" source="43" target="45" label="AST"><data key="type">AST</data></edge><edge id="59" source="43" target="46" label="NEXT_TOKEN"><data key="type">NEXT_TOKEN</data></edge></graph></graphml>