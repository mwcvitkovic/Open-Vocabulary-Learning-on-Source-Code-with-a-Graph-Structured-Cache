<?xml version="1.0" ?><graphml xmlns="http://graphml.graphdrawing.org/xmlns" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="http://graphml.graphdrawing.org/xmlns http://graphml.graphdrawing.org/xmlns/1.1/graphml.xsd"><key id="reference" for="node" attr.name="reference" attr.type="string"></key><key id="identifier" for="node" attr.name="identifier" attr.type="string"></key><key id="modifier" for="node" attr.name="modifier" attr.type="string"></key><key id="text" for="node" attr.name="text" attr.type="string"></key><key id="type" for="node" attr.name="type" attr.type="string"></key><key id="parentType" for="node" attr.name="parentType" attr.type="string"></key><key id="type" for="edge" attr.name="type" attr.type="string"></key><graph id="G" edgedefault="directed"><node id="44"><data key="reference"></data><data key="modifier">public</data><data key="text">/**
 * Return an existing instance of a certain contextual type or a null value.
 *
 * @param &lt;T&gt; the type of the contextual type
 * @param contextual the contextual type
 * @return the contextual instance, or a null value
 *
 * @throws ContextNotActiveException if the context is not active
 */
public &lt;T&gt; T get(Contextual&lt;T&gt; contextual);</data><data key="type">MethodDeclaration</data><data key="parentType">ClassOrInterfaceDeclaration</data></node><node id="45"><data key="reference"></data><data key="text">T</data><data key="type">TypeParameter</data><data key="parentType">MethodDeclaration</data></node><node id="46"><data key="reference">runtimeGenericType</data><data key="identifier">T</data><data key="text">T</data><data key="type">SimpleName</data><data key="parentType">TypeParameter</data></node><node id="47"><data key="reference">userDefinedMethodName</data><data key="identifier">get</data><data key="text">get</data><data key="type">SimpleName</data><data key="parentType">MethodDeclaration</data></node><node id="48"><data key="reference"></data><data key="modifier"></data><data key="text">Contextual&lt;T&gt; contextual</data><data key="type">Parameter</data><data key="parentType">MethodDeclaration</data></node><node id="49"><data key="reference"></data><data key="text">Contextual&lt;T&gt;</data><data key="type">ClassOrInterfaceType</data><data key="parentType">Parameter</data></node><node id="50"><data key="reference">nonQualifiedClassName</data><data key="identifier">Contextual</data><data key="text">Contextual</data><data key="type">SimpleName</data><data key="parentType">ClassOrInterfaceType</data></node><node id="51"><data key="reference"></data><data key="text">T</data><data key="type">ClassOrInterfaceType</data><data key="parentType">ClassOrInterfaceType</data></node><node id="52"><data key="reference">nonQualifiedClassName</data><data key="identifier">T</data><data key="text">T</data><data key="type">SimpleName</data><data key="parentType">ClassOrInterfaceType</data></node><node id="53"><data key="reference">javax.enterprise.context.spi.Contextual</data><data key="identifier">contextual</data><data key="text">contextual</data><data key="type">SimpleName</data><data key="parentType">Parameter</data></node><node id="10"><data key="reference"></data><data key="identifier">lang</data><data key="text">java.lang</data><data key="type">Name</data><data key="parentType">Name</data></node><node id="54"><data key="reference"></data><data key="text">T</data><data key="type">ClassOrInterfaceType</data><data key="parentType">MethodDeclaration</data></node><node id="11"><data key="reference"></data><data key="identifier">java</data><data key="text">java</data><data key="type">Name</data><data key="parentType">Name</data></node><node id="55"><data key="reference">nonQualifiedClassName</data><data key="identifier">T</data><data key="text">T</data><data key="type">SimpleName</data><data key="parentType">ClassOrInterfaceType</data></node><node id="12"><data key="reference"></data><data key="text">import javax.enterprise.context.ContextNotActiveException;
</data><data key="type">ImportDeclaration</data><data key="parentType">CompilationUnit</data></node><node id="56"><data key="reference"></data><data key="modifier">public</data><data key="text">/**
 * Determines if the context object is active.
 *
 * @return &lt;tt&gt;true&lt;/tt&gt; if the context is active, or &lt;tt&gt;false&lt;/tt&gt; otherwise.
 */
public boolean isActive();</data><data key="type">MethodDeclaration</data><data key="parentType">ClassOrInterfaceDeclaration</data></node><node id="13"><data key="reference"></data><data key="identifier">ContextNotActiveException</data><data key="text">javax.enterprise.context.ContextNotActiveException</data><data key="type">Name</data><data key="parentType">ImportDeclaration</data></node><node id="57"><data key="reference">userDefinedMethodName</data><data key="identifier">isActive</data><data key="text">isActive</data><data key="type">SimpleName</data><data key="parentType">MethodDeclaration</data></node><node id="14"><data key="reference"></data><data key="identifier">context</data><data key="text">javax.enterprise.context</data><data key="type">Name</data><data key="parentType">Name</data></node><node id="58"><data key="reference"></data><data key="identifier">boolean</data><data key="text">boolean</data><data key="type">PrimitiveType</data><data key="parentType">MethodDeclaration</data></node><node id="15"><data key="reference"></data><data key="identifier">enterprise</data><data key="text">javax.enterprise</data><data key="type">Name</data><data key="parentType">Name</data></node><node id="16"><data key="reference"></data><data key="identifier">javax</data><data key="text">javax</data><data key="type">Name</data><data key="parentType">Name</data></node><node id="17"><data key="reference"></data><data key="modifier">public,interface</data><data key="text">public interface Context {

    /**
     * Get the scope type of the context object.
     *
     * @return the scope
     */
    public Class&lt;? extends Annotation&gt; getScope();

    /**
     * Return an existing instance of certain contextual type or create a new instance by calling
     * {@link javax.enterprise.context.spi.Contextual#create(CreationalContext)} and return the new instance.
     *
     * @param &lt;T&gt; the type of contextual type
     * @param contextual the contextual type
     * @param creationalContext the context in which the new instance will be created
     * @return the contextual instance
     *
     * @throws ContextNotActiveException if the context is not active
     */
    public &lt;T&gt; T get(Contextual&lt;T&gt; contextual, CreationalContext&lt;T&gt; creationalContext);

    /**
     * Return an existing instance of a certain contextual type or a null value.
     *
     * @param &lt;T&gt; the type of the contextual type
     * @param contextual the contextual type
     * @return the contextual instance, or a null value
     *
     * @throws ContextNotActiveException if the context is not active
     */
    public &lt;T&gt; T get(Contextual&lt;T&gt; contextual);

    /**
     * Determines if the context object is active.
     *
     * @return &lt;tt&gt;true&lt;/tt&gt; if the context is active, or &lt;tt&gt;false&lt;/tt&gt; otherwise.
     */
    public boolean isActive();
}</data><data key="type">ClassOrInterfaceDeclaration</data><data key="parentType">CompilationUnit</data></node><node id="18"><data key="reference"></data><data key="identifier">Context</data><data key="text">Context</data><data key="type">SimpleName</data><data key="parentType">ClassOrInterfaceDeclaration</data></node><node id="19"><data key="reference"></data><data key="modifier">public</data><data key="text">/**
 * Get the scope type of the context object.
 *
 * @return the scope
 */
public Class&lt;? extends Annotation&gt; getScope();</data><data key="type">MethodDeclaration</data><data key="parentType">ClassOrInterfaceDeclaration</data></node><node id="1"><data key="reference"></data><data key="text">/*
 * JBoss, Home of Professional Open Source
 * Copyright 2010, Red Hat, Inc., and individual contributors
 * by the @authors tag. See the copyright.txt in the distribution for a
 * full listing of individual contributors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,  
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package javax.enterprise.context.spi;

import java.lang.annotation.Annotation;
import javax.enterprise.context.ContextNotActiveException;

public interface Context {

    /**
     * Get the scope type of the context object.
     *
     * @return the scope
     */
    public Class&lt;? extends Annotation&gt; getScope();

    /**
     * Return an existing instance of certain contextual type or create a new instance by calling
     * {@link javax.enterprise.context.spi.Contextual#create(CreationalContext)} and return the new instance.
     *
     * @param &lt;T&gt; the type of contextual type
     * @param contextual the contextual type
     * @param creationalContext the context in which the new instance will be created
     * @return the contextual instance
     *
     * @throws ContextNotActiveException if the context is not active
     */
    public &lt;T&gt; T get(Contextual&lt;T&gt; contextual, CreationalContext&lt;T&gt; creationalContext);

    /**
     * Return an existing instance of a certain contextual type or a null value.
     *
     * @param &lt;T&gt; the type of the contextual type
     * @param contextual the contextual type
     * @return the contextual instance, or a null value
     *
     * @throws ContextNotActiveException if the context is not active
     */
    public &lt;T&gt; T get(Contextual&lt;T&gt; contextual);

    /**
     * Determines if the context object is active.
     *
     * @return &lt;tt&gt;true&lt;/tt&gt; if the context is active, or &lt;tt&gt;false&lt;/tt&gt; otherwise.
     */
    public boolean isActive();
}
</data><data key="type">CompilationUnit</data></node><node id="2"><data key="reference"></data><data key="text">package javax.enterprise.context.spi;

</data><data key="type">PackageDeclaration</data><data key="parentType">CompilationUnit</data></node><node id="3"><data key="reference"></data><data key="identifier">spi</data><data key="text">javax.enterprise.context.spi</data><data key="type">Name</data><data key="parentType">PackageDeclaration</data></node><node id="4"><data key="reference"></data><data key="identifier">context</data><data key="text">javax.enterprise.context</data><data key="type">Name</data><data key="parentType">Name</data></node><node id="5"><data key="reference"></data><data key="identifier">enterprise</data><data key="text">javax.enterprise</data><data key="type">Name</data><data key="parentType">Name</data></node><node id="6"><data key="reference"></data><data key="identifier">javax</data><data key="text">javax</data><data key="type">Name</data><data key="parentType">Name</data></node><node id="7"><data key="reference"></data><data key="text">import java.lang.annotation.Annotation;
</data><data key="type">ImportDeclaration</data><data key="parentType">CompilationUnit</data></node><node id="8"><data key="reference"></data><data key="identifier">Annotation</data><data key="text">java.lang.annotation.Annotation</data><data key="type">Name</data><data key="parentType">ImportDeclaration</data></node><node id="9"><data key="reference"></data><data key="identifier">annotation</data><data key="text">java.lang.annotation</data><data key="type">Name</data><data key="parentType">Name</data></node><node id="20"><data key="reference">userDefinedMethodName</data><data key="identifier">getScope</data><data key="text">getScope</data><data key="type">SimpleName</data><data key="parentType">MethodDeclaration</data></node><node id="21"><data key="reference"></data><data key="text">Class&lt;? extends Annotation&gt;</data><data key="type">ClassOrInterfaceType</data><data key="parentType">MethodDeclaration</data></node><node id="22"><data key="reference">nonQualifiedClassName</data><data key="identifier">Class</data><data key="text">Class</data><data key="type">SimpleName</data><data key="parentType">ClassOrInterfaceType</data></node><node id="23"><data key="reference"></data><data key="text">? extends Annotation</data><data key="type">WildcardType</data><data key="parentType">ClassOrInterfaceType</data></node><node id="24"><data key="reference"></data><data key="text">Annotation</data><data key="type">ClassOrInterfaceType</data><data key="parentType">WildcardType</data></node><node id="25"><data key="reference">nonQualifiedClassName</data><data key="identifier">Annotation</data><data key="text">Annotation</data><data key="type">SimpleName</data><data key="parentType">ClassOrInterfaceType</data></node><node id="26"><data key="reference"></data><data key="modifier">public</data><data key="text">/**
 * Return an existing instance of certain contextual type or create a new instance by calling
 * {@link javax.enterprise.context.spi.Contextual#create(CreationalContext)} and return the new instance.
 *
 * @param &lt;T&gt; the type of contextual type
 * @param contextual the contextual type
 * @param creationalContext the context in which the new instance will be created
 * @return the contextual instance
 *
 * @throws ContextNotActiveException if the context is not active
 */
public &lt;T&gt; T get(Contextual&lt;T&gt; contextual, CreationalContext&lt;T&gt; creationalContext);</data><data key="type">MethodDeclaration</data><data key="parentType">ClassOrInterfaceDeclaration</data></node><node id="27"><data key="reference"></data><data key="text">T</data><data key="type">TypeParameter</data><data key="parentType">MethodDeclaration</data></node><node id="28"><data key="reference">runtimeGenericType</data><data key="identifier">T</data><data key="text">T</data><data key="type">SimpleName</data><data key="parentType">TypeParameter</data></node><node id="29"><data key="reference">userDefinedMethodName</data><data key="identifier">get</data><data key="text">get</data><data key="type">SimpleName</data><data key="parentType">MethodDeclaration</data></node><node id="30"><data key="reference"></data><data key="modifier"></data><data key="text">Contextual&lt;T&gt; contextual</data><data key="type">Parameter</data><data key="parentType">MethodDeclaration</data></node><node id="31"><data key="reference"></data><data key="text">Contextual&lt;T&gt;</data><data key="type">ClassOrInterfaceType</data><data key="parentType">Parameter</data></node><node id="32"><data key="reference">nonQualifiedClassName</data><data key="identifier">Contextual</data><data key="text">Contextual</data><data key="type">SimpleName</data><data key="parentType">ClassOrInterfaceType</data></node><node id="33"><data key="reference"></data><data key="text">T</data><data key="type">ClassOrInterfaceType</data><data key="parentType">ClassOrInterfaceType</data></node><node id="34"><data key="reference">nonQualifiedClassName</data><data key="identifier">T</data><data key="text">T</data><data key="type">SimpleName</data><data key="parentType">ClassOrInterfaceType</data></node><node id="35"><data key="reference">javax.enterprise.context.spi.Contextual</data><data key="identifier">contextual</data><data key="text">contextual</data><data key="type">SimpleName</data><data key="parentType">Parameter</data></node><node id="36"><data key="reference"></data><data key="modifier"></data><data key="text">CreationalContext&lt;T&gt; creationalContext</data><data key="type">Parameter</data><data key="parentType">MethodDeclaration</data></node><node id="37"><data key="reference"></data><data key="text">CreationalContext&lt;T&gt;</data><data key="type">ClassOrInterfaceType</data><data key="parentType">Parameter</data></node><node id="38"><data key="reference">nonQualifiedClassName</data><data key="identifier">CreationalContext</data><data key="text">CreationalContext</data><data key="type">SimpleName</data><data key="parentType">ClassOrInterfaceType</data></node><node id="39"><data key="reference"></data><data key="text">T</data><data key="type">ClassOrInterfaceType</data><data key="parentType">ClassOrInterfaceType</data></node><node id="40"><data key="reference">nonQualifiedClassName</data><data key="identifier">T</data><data key="text">T</data><data key="type">SimpleName</data><data key="parentType">ClassOrInterfaceType</data></node><node id="41"><data key="reference">javax.enterprise.context.spi.CreationalContext</data><data key="identifier">creationalContext</data><data key="text">creationalContext</data><data key="type">SimpleName</data><data key="parentType">Parameter</data></node><node id="42"><data key="reference"></data><data key="text">T</data><data key="type">ClassOrInterfaceType</data><data key="parentType">MethodDeclaration</data></node><node id="43"><data key="reference">nonQualifiedClassName</data><data key="identifier">T</data><data key="text">T</data><data key="type">SimpleName</data><data key="parentType">ClassOrInterfaceType</data></node><edge id="60" source="44" target="45" label="AST"><data key="type">AST</data></edge><edge id="63" source="44" target="47" label="AST"><data key="type">AST</data></edge><edge id="74" source="44" target="54" label="AST"><data key="type">AST</data></edge><edge id="65" source="44" target="48" label="AST"><data key="type">AST</data></edge><edge id="59" source="44" target="56" label="NEXT_TOKEN"><data key="type">NEXT_TOKEN</data></edge><edge id="62" source="45" target="46" label="AST"><data key="type">AST</data></edge><edge id="61" source="45" target="47" label="NEXT_TOKEN"><data key="type">NEXT_TOKEN</data></edge><edge id="64" source="47" target="48" label="NEXT_TOKEN"><data key="type">NEXT_TOKEN</data></edge><edge id="67" source="48" target="49" label="AST"><data key="type">AST</data></edge><edge id="73" source="48" target="53" label="AST"><data key="type">AST</data></edge><edge id="66" source="48" target="54" label="NEXT_TOKEN"><data key="type">NEXT_TOKEN</data></edge><edge id="69" source="49" target="50" label="AST"><data key="type">AST</data></edge><edge id="71" source="49" target="51" label="AST"><data key="type">AST</data></edge><edge id="68" source="49" target="53" label="NEXT_TOKEN"><data key="type">NEXT_TOKEN</data></edge><edge id="70" source="50" target="51" label="NEXT_TOKEN"><data key="type">NEXT_TOKEN</data></edge><edge id="72" source="51" target="52" label="AST"><data key="type">AST</data></edge><edge id="11" source="10" target="11" label="AST"><data key="type">AST</data></edge><edge id="75" source="54" target="55" label="AST"><data key="type">AST</data></edge><edge id="14" source="12" target="13" label="AST"><data key="type">AST</data></edge><edge id="13" source="12" target="17" label="NEXT_TOKEN"><data key="type">NEXT_TOKEN</data></edge><edge id="77" source="56" target="57" label="AST"><data key="type">AST</data></edge><edge id="79" source="56" target="58" label="AST"><data key="type">AST</data></edge><edge id="15" source="13" target="14" label="AST"><data key="type">AST</data></edge><edge id="78" source="57" target="58" label="NEXT_TOKEN"><data key="type">NEXT_TOKEN</data></edge><edge id="16" source="14" target="15" label="AST"><data key="type">AST</data></edge><edge id="17" source="15" target="16" label="AST"><data key="type">AST</data></edge><edge id="58" source="17" target="44" label="AST"><data key="type">AST</data></edge><edge id="19" source="17" target="18" label="AST"><data key="type">AST</data></edge><edge id="31" source="17" target="26" label="AST"><data key="type">AST</data></edge><edge id="21" source="17" target="19" label="AST"><data key="type">AST</data></edge><edge id="76" source="17" target="56" label="AST"><data key="type">AST</data></edge><edge id="20" source="18" target="19" label="NEXT_TOKEN"><data key="type">NEXT_TOKEN</data></edge><edge id="23" source="19" target="20" label="AST"><data key="type">AST</data></edge><edge id="25" source="19" target="21" label="AST"><data key="type">AST</data></edge><edge id="22" source="19" target="26" label="NEXT_TOKEN"><data key="type">NEXT_TOKEN</data></edge><edge id="0" source="1" target="2" label="AST"><data key="type">AST</data></edge><edge id="12" source="1" target="12" label="AST"><data key="type">AST</data></edge><edge id="6" source="1" target="7" label="AST"><data key="type">AST</data></edge><edge id="18" source="1" target="17" label="AST"><data key="type">AST</data></edge><edge id="2" source="2" target="3" label="AST"><data key="type">AST</data></edge><edge id="1" source="2" target="7" label="NEXT_TOKEN"><data key="type">NEXT_TOKEN</data></edge><edge id="3" source="3" target="4" label="AST"><data key="type">AST</data></edge><edge id="4" source="4" target="5" label="AST"><data key="type">AST</data></edge><edge id="5" source="5" target="6" label="AST"><data key="type">AST</data></edge><edge id="8" source="7" target="8" label="AST"><data key="type">AST</data></edge><edge id="7" source="7" target="12" label="NEXT_TOKEN"><data key="type">NEXT_TOKEN</data></edge><edge id="9" source="8" target="9" label="AST"><data key="type">AST</data></edge><edge id="10" source="9" target="10" label="AST"><data key="type">AST</data></edge><edge id="24" source="20" target="21" label="NEXT_TOKEN"><data key="type">NEXT_TOKEN</data></edge><edge id="26" source="21" target="22" label="AST"><data key="type">AST</data></edge><edge id="28" source="21" target="23" label="AST"><data key="type">AST</data></edge><edge id="27" source="22" target="23" label="NEXT_TOKEN"><data key="type">NEXT_TOKEN</data></edge><edge id="29" source="23" target="24" label="AST"><data key="type">AST</data></edge><edge id="30" source="24" target="25" label="AST"><data key="type">AST</data></edge><edge id="33" source="26" target="27" label="AST"><data key="type">AST</data></edge><edge id="56" source="26" target="42" label="AST"><data key="type">AST</data></edge><edge id="36" source="26" target="29" label="AST"><data key="type">AST</data></edge><edge id="47" source="26" target="36" label="AST"><data key="type">AST</data></edge><edge id="38" source="26" target="30" label="AST"><data key="type">AST</data></edge><edge id="32" source="26" target="44" label="NEXT_TOKEN"><data key="type">NEXT_TOKEN</data></edge><edge id="35" source="27" target="28" label="AST"><data key="type">AST</data></edge><edge id="34" source="27" target="29" label="NEXT_TOKEN"><data key="type">NEXT_TOKEN</data></edge><edge id="37" source="29" target="30" label="NEXT_TOKEN"><data key="type">NEXT_TOKEN</data></edge><edge id="46" source="30" target="35" label="AST"><data key="type">AST</data></edge><edge id="40" source="30" target="31" label="AST"><data key="type">AST</data></edge><edge id="39" source="30" target="36" label="NEXT_TOKEN"><data key="type">NEXT_TOKEN</data></edge><edge id="44" source="31" target="33" label="AST"><data key="type">AST</data></edge><edge id="42" source="31" target="32" label="AST"><data key="type">AST</data></edge><edge id="41" source="31" target="35" label="NEXT_TOKEN"><data key="type">NEXT_TOKEN</data></edge><edge id="43" source="32" target="33" label="NEXT_TOKEN"><data key="type">NEXT_TOKEN</data></edge><edge id="45" source="33" target="34" label="AST"><data key="type">AST</data></edge><edge id="55" source="36" target="41" label="AST"><data key="type">AST</data></edge><edge id="49" source="36" target="37" label="AST"><data key="type">AST</data></edge><edge id="48" source="36" target="42" label="NEXT_TOKEN"><data key="type">NEXT_TOKEN</data></edge><edge id="51" source="37" target="38" label="AST"><data key="type">AST</data></edge><edge id="53" source="37" target="39" label="AST"><data key="type">AST</data></edge><edge id="50" source="37" target="41" label="NEXT_TOKEN"><data key="type">NEXT_TOKEN</data></edge><edge id="52" source="38" target="39" label="NEXT_TOKEN"><data key="type">NEXT_TOKEN</data></edge><edge id="54" source="39" target="40" label="AST"><data key="type">AST</data></edge><edge id="57" source="42" target="43" label="AST"><data key="type">AST</data></edge></graph></graphml>