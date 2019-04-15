<?xml version="1.0" ?><graphml xmlns="http://graphml.graphdrawing.org/xmlns" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="http://graphml.graphdrawing.org/xmlns http://graphml.graphdrawing.org/xmlns/1.1/graphml.xsd"><key id="reference" for="node" attr.name="reference" attr.type="string"></key><key id="identifier" for="node" attr.name="identifier" attr.type="string"></key><key id="modifier" for="node" attr.name="modifier" attr.type="string"></key><key id="text" for="node" attr.name="text" attr.type="string"></key><key id="type" for="node" attr.name="type" attr.type="string"></key><key id="parentType" for="node" attr.name="parentType" attr.type="string"></key><key id="type" for="edge" attr.name="type" attr.type="string"></key><graph id="G" edgedefault="directed"><node id="88"><data key="reference">userDefinedMethodName</data><data key="identifier">getKey</data><data key="text">getKey</data><data key="type">SimpleName</data><data key="parentType">MethodDeclaration</data></node><node id="89"><data key="reference"></data><data key="text">K</data><data key="type">ClassOrInterfaceType</data><data key="parentType">MethodDeclaration</data></node><node id="110"><data key="reference">genericType</data><data key="identifier">obj</data><data key="text">obj</data><data key="type">SimpleName</data><data key="parentType">Parameter</data></node><node id="111"><data key="reference"></data><data key="text">V</data><data key="type">ClassOrInterfaceType</data><data key="parentType">MethodDeclaration</data></node><node id="112"><data key="reference">nonQualifiedClassName</data><data key="identifier">V</data><data key="text">V</data><data key="type">SimpleName</data><data key="parentType">ClassOrInterfaceType</data></node><node id="113"><data key="reference"></data><data key="text">{
    return iterator.setValue(obj);
}</data><data key="type">BlockStmt</data><data key="parentType">MethodDeclaration</data></node><node id="114"><data key="reference"></data><data key="text">return iterator.setValue(obj);</data><data key="type">ReturnStmt</data><data key="parentType">BlockStmt</data></node><node id="115"><data key="reference"></data><data key="text">iterator.setValue(obj)</data><data key="type">MethodCallExpr</data><data key="parentType">ReturnStmt</data></node><node id="116"><data key="reference">org.apache.commons.collections4.MapIterator</data><data key="identifier">iterator</data><data key="text">iterator</data><data key="type">SimpleName</data><data key="parentType">NameExpr</data></node><node id="90"><data key="reference">nonQualifiedClassName</data><data key="identifier">K</data><data key="text">K</data><data key="type">SimpleName</data><data key="parentType">ClassOrInterfaceType</data></node><node id="117"><data key="reference">otherMethodCall</data><data key="identifier">setValue</data><data key="text">setValue</data><data key="type">SimpleName</data><data key="parentType">MethodCallExpr</data></node><node id="91"><data key="reference"></data><data key="text">{
    return iterator.getKey();
}</data><data key="type">BlockStmt</data><data key="parentType">MethodDeclaration</data></node><node id="118"><data key="reference">genericType</data><data key="identifier">obj</data><data key="text">obj</data><data key="type">SimpleName</data><data key="parentType">NameExpr</data></node><node id="92"><data key="reference"></data><data key="text">return iterator.getKey();</data><data key="type">ReturnStmt</data><data key="parentType">BlockStmt</data></node><node id="119"><data key="reference"></data><data key="text">K</data><data key="type">TypeParameter</data><data key="parentType">ClassOrInterfaceDeclaration</data></node><node id="93"><data key="reference"></data><data key="text">iterator.getKey()</data><data key="type">MethodCallExpr</data><data key="parentType">ReturnStmt</data></node><node id="94"><data key="reference">org.apache.commons.collections4.MapIterator</data><data key="identifier">iterator</data><data key="text">iterator</data><data key="type">SimpleName</data><data key="parentType">NameExpr</data></node><node id="95"><data key="reference">otherMethodCall</data><data key="identifier">getKey</data><data key="text">getKey</data><data key="type">SimpleName</data><data key="parentType">MethodCallExpr</data></node><node id="96"><data key="reference"></data><data key="modifier">public</data><data key="text">/**
 * {@inheritDoc}
 */
public V getValue() {
    return iterator.getValue();
}</data><data key="type">MethodDeclaration</data><data key="parentType">ClassOrInterfaceDeclaration</data></node><node id="97"><data key="reference">userDefinedMethodName</data><data key="identifier">getValue</data><data key="text">getValue</data><data key="type">SimpleName</data><data key="parentType">MethodDeclaration</data></node><node id="10"><data key="reference"></data><data key="identifier">collections4</data><data key="text">org.apache.commons.collections4</data><data key="type">Name</data><data key="parentType">Name</data></node><node id="98"><data key="reference"></data><data key="text">V</data><data key="type">ClassOrInterfaceType</data><data key="parentType">MethodDeclaration</data></node><node id="11"><data key="reference"></data><data key="identifier">commons</data><data key="text">org.apache.commons</data><data key="type">Name</data><data key="parentType">Name</data></node><node id="99"><data key="reference">nonQualifiedClassName</data><data key="identifier">V</data><data key="text">V</data><data key="type">SimpleName</data><data key="parentType">ClassOrInterfaceType</data></node><node id="12"><data key="reference"></data><data key="identifier">apache</data><data key="text">org.apache</data><data key="type">Name</data><data key="parentType">Name</data></node><node id="13"><data key="reference"></data><data key="identifier">org</data><data key="text">org</data><data key="type">Name</data><data key="parentType">Name</data></node><node id="14"><data key="reference"></data><data key="modifier">public</data><data key="text">/**
 * Provides basic behaviour for decorating a map iterator with extra functionality.
 * &lt;p&gt;
 * All methods are forwarded to the decorated map iterator.
 *
 * @since 3.0
 * @version $Id: AbstractMapIteratorDecorator.java 1686855 2015-06-22 13:00:27Z tn $
 */
public class AbstractMapIteratorDecorator&lt;K, V&gt; implements MapIterator&lt;K, V&gt; {

    /**
     * The iterator being decorated
     */
    private final MapIterator&lt;K, V&gt; iterator;

    /**
     * Constructor that decorates the specified iterator.
     *
     * @param iterator  the iterator to decorate, must not be null
     * @throws NullPointerException if the iterator is null
     */
    public AbstractMapIteratorDecorator(final MapIterator&lt;K, V&gt; iterator) {
        super();
        if (iterator == null) {
            throw new NullPointerException("MapIterator must not be null");
        }
        this.iterator = iterator;
    }

    /**
     * Gets the iterator being decorated.
     *
     * @return the decorated iterator
     */
    protected MapIterator&lt;K, V&gt; getMapIterator() {
        return iterator;
    }

    // -----------------------------------------------------------------------
    /**
     * {@inheritDoc}
     */
    public boolean hasNext() {
        return iterator.hasNext();
    }

    /**
     * {@inheritDoc}
     */
    public K next() {
        return iterator.next();
    }

    /**
     * {@inheritDoc}
     */
    public void remove() {
        iterator.remove();
    }

    /**
     * {@inheritDoc}
     */
    public K getKey() {
        return iterator.getKey();
    }

    /**
     * {@inheritDoc}
     */
    public V getValue() {
        return iterator.getValue();
    }

    /**
     * {@inheritDoc}
     */
    public V setValue(final V obj) {
        return iterator.setValue(obj);
    }
}</data><data key="type">ClassOrInterfaceDeclaration</data><data key="parentType">CompilationUnit</data></node><node id="15"><data key="reference">java.lang.Object,org.apache.commons.collections4.MapIterator,java.util.Iterator</data><data key="identifier">AbstractMapIteratorDecorator</data><data key="text">AbstractMapIteratorDecorator</data><data key="type">SimpleName</data><data key="parentType">ClassOrInterfaceDeclaration</data></node><node id="16"><data key="reference"></data><data key="modifier">private,final</data><data key="text">/**
 * The iterator being decorated
 */
private final MapIterator&lt;K, V&gt; iterator;</data><data key="type">FieldDeclaration</data><data key="parentType">ClassOrInterfaceDeclaration</data></node><node id="17"><data key="reference"></data><data key="text">iterator</data><data key="type">VariableDeclarator</data><data key="parentType">FieldDeclaration</data></node><node id="18"><data key="reference"></data><data key="text">MapIterator&lt;K, V&gt;</data><data key="type">ClassOrInterfaceType</data><data key="parentType">VariableDeclarator</data></node><node id="19"><data key="reference">nonQualifiedClassName</data><data key="identifier">MapIterator</data><data key="text">MapIterator</data><data key="type">SimpleName</data><data key="parentType">ClassOrInterfaceType</data></node><node id="120"><data key="reference">runtimeGenericType</data><data key="identifier">K</data><data key="text">K</data><data key="type">SimpleName</data><data key="parentType">TypeParameter</data></node><node id="121"><data key="reference"></data><data key="text">V</data><data key="type">TypeParameter</data><data key="parentType">ClassOrInterfaceDeclaration</data></node><node id="1"><data key="reference"></data><data key="text">/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.apache.commons.collections4.iterators;

import org.apache.commons.collections4.MapIterator;

/**
 * Provides basic behaviour for decorating a map iterator with extra functionality.
 * &lt;p&gt;
 * All methods are forwarded to the decorated map iterator.
 *
 * @since 3.0
 * @version $Id: AbstractMapIteratorDecorator.java 1686855 2015-06-22 13:00:27Z tn $
 */
public class AbstractMapIteratorDecorator&lt;K, V&gt; implements MapIterator&lt;K, V&gt; {

    /**
     * The iterator being decorated
     */
    private final MapIterator&lt;K, V&gt; iterator;

    /**
     * Constructor that decorates the specified iterator.
     *
     * @param iterator  the iterator to decorate, must not be null
     * @throws NullPointerException if the iterator is null
     */
    public AbstractMapIteratorDecorator(final MapIterator&lt;K, V&gt; iterator) {
        super();
        if (iterator == null) {
            throw new NullPointerException("MapIterator must not be null");
        }
        this.iterator = iterator;
    }

    /**
     * Gets the iterator being decorated.
     *
     * @return the decorated iterator
     */
    protected MapIterator&lt;K, V&gt; getMapIterator() {
        return iterator;
    }

    // -----------------------------------------------------------------------
    /**
     * {@inheritDoc}
     */
    public boolean hasNext() {
        return iterator.hasNext();
    }

    /**
     * {@inheritDoc}
     */
    public K next() {
        return iterator.next();
    }

    /**
     * {@inheritDoc}
     */
    public void remove() {
        iterator.remove();
    }

    /**
     * {@inheritDoc}
     */
    public K getKey() {
        return iterator.getKey();
    }

    /**
     * {@inheritDoc}
     */
    public V getValue() {
        return iterator.getValue();
    }

    /**
     * {@inheritDoc}
     */
    public V setValue(final V obj) {
        return iterator.setValue(obj);
    }
}
</data><data key="type">CompilationUnit</data></node><node id="122"><data key="reference">runtimeGenericType</data><data key="identifier">V</data><data key="text">V</data><data key="type">SimpleName</data><data key="parentType">TypeParameter</data></node><node id="2"><data key="reference"></data><data key="text">package org.apache.commons.collections4.iterators;

</data><data key="type">PackageDeclaration</data><data key="parentType">CompilationUnit</data></node><node id="123"><data key="reference"></data><data key="text">MapIterator&lt;K, V&gt;</data><data key="type">ClassOrInterfaceType</data><data key="parentType">ClassOrInterfaceDeclaration</data></node><node id="3"><data key="reference"></data><data key="identifier">iterators</data><data key="text">org.apache.commons.collections4.iterators</data><data key="type">Name</data><data key="parentType">PackageDeclaration</data></node><node id="124"><data key="reference">nonQualifiedClassName</data><data key="identifier">MapIterator</data><data key="text">MapIterator</data><data key="type">SimpleName</data><data key="parentType">ClassOrInterfaceType</data></node><node id="4"><data key="reference"></data><data key="identifier">collections4</data><data key="text">org.apache.commons.collections4</data><data key="type">Name</data><data key="parentType">Name</data></node><node id="125"><data key="reference"></data><data key="text">K</data><data key="type">ClassOrInterfaceType</data><data key="parentType">ClassOrInterfaceType</data></node><node id="5"><data key="reference"></data><data key="identifier">commons</data><data key="text">org.apache.commons</data><data key="type">Name</data><data key="parentType">Name</data></node><node id="126"><data key="reference">nonQualifiedClassName</data><data key="identifier">K</data><data key="text">K</data><data key="type">SimpleName</data><data key="parentType">ClassOrInterfaceType</data></node><node id="6"><data key="reference"></data><data key="identifier">apache</data><data key="text">org.apache</data><data key="type">Name</data><data key="parentType">Name</data></node><node id="127"><data key="reference"></data><data key="text">V</data><data key="type">ClassOrInterfaceType</data><data key="parentType">ClassOrInterfaceType</data></node><node id="7"><data key="reference"></data><data key="identifier">org</data><data key="text">org</data><data key="type">Name</data><data key="parentType">Name</data></node><node id="128"><data key="reference">nonQualifiedClassName</data><data key="identifier">V</data><data key="text">V</data><data key="type">SimpleName</data><data key="parentType">ClassOrInterfaceType</data></node><node id="8"><data key="reference"></data><data key="text">import org.apache.commons.collections4.MapIterator;
</data><data key="type">ImportDeclaration</data><data key="parentType">CompilationUnit</data></node><node id="9"><data key="reference"></data><data key="identifier">MapIterator</data><data key="text">org.apache.commons.collections4.MapIterator</data><data key="type">Name</data><data key="parentType">ImportDeclaration</data></node><node id="20"><data key="reference"></data><data key="text">K</data><data key="type">ClassOrInterfaceType</data><data key="parentType">ClassOrInterfaceType</data></node><node id="21"><data key="reference">nonQualifiedClassName</data><data key="identifier">K</data><data key="text">K</data><data key="type">SimpleName</data><data key="parentType">ClassOrInterfaceType</data></node><node id="22"><data key="reference"></data><data key="text">V</data><data key="type">ClassOrInterfaceType</data><data key="parentType">ClassOrInterfaceType</data></node><node id="23"><data key="reference">nonQualifiedClassName</data><data key="identifier">V</data><data key="text">V</data><data key="type">SimpleName</data><data key="parentType">ClassOrInterfaceType</data></node><node id="24"><data key="reference">org.apache.commons.collections4.MapIterator</data><data key="identifier">iterator</data><data key="text">iterator</data><data key="type">SimpleName</data><data key="parentType">VariableDeclarator</data></node><node id="25"><data key="reference"></data><data key="modifier">public</data><data key="text">/**
 * Constructor that decorates the specified iterator.
 *
 * @param iterator  the iterator to decorate, must not be null
 * @throws NullPointerException if the iterator is null
 */
public AbstractMapIteratorDecorator(final MapIterator&lt;K, V&gt; iterator) {
    super();
    if (iterator == null) {
        throw new NullPointerException("MapIterator must not be null");
    }
    this.iterator = iterator;
}</data><data key="type">ConstructorDeclaration</data><data key="parentType">ClassOrInterfaceDeclaration</data></node><node id="26"><data key="reference">userDefinedMethodName</data><data key="identifier">AbstractMapIteratorDecorator</data><data key="text">AbstractMapIteratorDecorator</data><data key="type">SimpleName</data><data key="parentType">ConstructorDeclaration</data></node><node id="27"><data key="reference"></data><data key="modifier">final</data><data key="text">final MapIterator&lt;K, V&gt; iterator</data><data key="type">Parameter</data><data key="parentType">ConstructorDeclaration</data></node><node id="28"><data key="reference"></data><data key="text">MapIterator&lt;K, V&gt;</data><data key="type">ClassOrInterfaceType</data><data key="parentType">Parameter</data></node><node id="29"><data key="reference">nonQualifiedClassName</data><data key="identifier">MapIterator</data><data key="text">MapIterator</data><data key="type">SimpleName</data><data key="parentType">ClassOrInterfaceType</data></node><node id="30"><data key="reference"></data><data key="text">K</data><data key="type">ClassOrInterfaceType</data><data key="parentType">ClassOrInterfaceType</data></node><node id="31"><data key="reference">nonQualifiedClassName</data><data key="identifier">K</data><data key="text">K</data><data key="type">SimpleName</data><data key="parentType">ClassOrInterfaceType</data></node><node id="32"><data key="reference"></data><data key="text">V</data><data key="type">ClassOrInterfaceType</data><data key="parentType">ClassOrInterfaceType</data></node><node id="33"><data key="reference">nonQualifiedClassName</data><data key="identifier">V</data><data key="text">V</data><data key="type">SimpleName</data><data key="parentType">ClassOrInterfaceType</data></node><node id="34"><data key="reference">org.apache.commons.collections4.MapIterator</data><data key="identifier">iterator</data><data key="text">iterator</data><data key="type">SimpleName</data><data key="parentType">Parameter</data></node><node id="35"><data key="reference"></data><data key="text">{
    super();
    if (iterator == null) {
        throw new NullPointerException("MapIterator must not be null");
    }
    this.iterator = iterator;
}</data><data key="type">BlockStmt</data><data key="parentType">ConstructorDeclaration</data></node><node id="36"><data key="reference"></data><data key="text">super();</data><data key="type">ExplicitConstructorInvocationStmt</data><data key="parentType">BlockStmt</data></node><node id="37"><data key="reference"></data><data key="text">if (iterator == null) {
    throw new NullPointerException("MapIterator must not be null");
}</data><data key="type">IfStmt</data><data key="parentType">BlockStmt</data></node><node id="38"><data key="reference"></data><data key="text">iterator == null</data><data key="type">BinaryExpr</data><data key="parentType">IfStmt</data></node><node id="39"><data key="reference">org.apache.commons.collections4.MapIterator</data><data key="identifier">iterator</data><data key="text">iterator</data><data key="type">SimpleName</data><data key="parentType">NameExpr</data></node><node id="40"><data key="reference"></data><data key="text">null</data><data key="type">NullLiteralExpr</data><data key="parentType">BinaryExpr</data></node><node id="41"><data key="reference"></data><data key="text">{
    throw new NullPointerException("MapIterator must not be null");
}</data><data key="type">BlockStmt</data><data key="parentType">IfStmt</data></node><node id="42"><data key="reference"></data><data key="text">throw new NullPointerException("MapIterator must not be null");</data><data key="type">ThrowStmt</data><data key="parentType">BlockStmt</data></node><node id="43"><data key="reference"></data><data key="text">new NullPointerException("MapIterator must not be null")</data><data key="type">ObjectCreationExpr</data><data key="parentType">ThrowStmt</data></node><node id="44"><data key="reference"></data><data key="text">NullPointerException</data><data key="type">ClassOrInterfaceType</data><data key="parentType">ObjectCreationExpr</data></node><node id="45"><data key="reference">nonQualifiedClassName</data><data key="identifier">NullPointerException</data><data key="text">NullPointerException</data><data key="type">SimpleName</data><data key="parentType">ClassOrInterfaceType</data></node><node id="46"><data key="reference"></data><data key="identifier">MapIterator must not be null</data><data key="text">"MapIterator must not be null"</data><data key="type">StringLiteralExpr</data><data key="parentType">ObjectCreationExpr</data></node><node id="47"><data key="reference"></data><data key="identifier">=</data><data key="text">this.iterator = iterator</data><data key="type">AssignExpr</data><data key="parentType">ExpressionStmt</data></node><node id="48"><data key="reference"></data><data key="text">this.iterator</data><data key="type">FieldAccessExpr</data><data key="parentType">AssignExpr</data></node><node id="49"><data key="reference"></data><data key="text">this</data><data key="type">ThisExpr</data><data key="parentType">FieldAccessExpr</data></node><node id="50"><data key="reference">org.apache.commons.collections4.MapIterator</data><data key="identifier">iterator</data><data key="text">iterator</data><data key="type">SimpleName</data><data key="parentType">FieldAccessExpr</data></node><node id="51"><data key="reference">org.apache.commons.collections4.MapIterator</data><data key="identifier">iterator</data><data key="text">iterator</data><data key="type">SimpleName</data><data key="parentType">NameExpr</data></node><node id="52"><data key="reference"></data><data key="modifier">protected</data><data key="text">/**
 * Gets the iterator being decorated.
 *
 * @return the decorated iterator
 */
protected MapIterator&lt;K, V&gt; getMapIterator() {
    return iterator;
}</data><data key="type">MethodDeclaration</data><data key="parentType">ClassOrInterfaceDeclaration</data></node><node id="53"><data key="reference">userDefinedMethodName</data><data key="identifier">getMapIterator</data><data key="text">getMapIterator</data><data key="type">SimpleName</data><data key="parentType">MethodDeclaration</data></node><node id="54"><data key="reference"></data><data key="text">MapIterator&lt;K, V&gt;</data><data key="type">ClassOrInterfaceType</data><data key="parentType">MethodDeclaration</data></node><node id="55"><data key="reference">nonQualifiedClassName</data><data key="identifier">MapIterator</data><data key="text">MapIterator</data><data key="type">SimpleName</data><data key="parentType">ClassOrInterfaceType</data></node><node id="56"><data key="reference"></data><data key="text">K</data><data key="type">ClassOrInterfaceType</data><data key="parentType">ClassOrInterfaceType</data></node><node id="57"><data key="reference">nonQualifiedClassName</data><data key="identifier">K</data><data key="text">K</data><data key="type">SimpleName</data><data key="parentType">ClassOrInterfaceType</data></node><node id="58"><data key="reference"></data><data key="text">V</data><data key="type">ClassOrInterfaceType</data><data key="parentType">ClassOrInterfaceType</data></node><node id="59"><data key="reference">nonQualifiedClassName</data><data key="identifier">V</data><data key="text">V</data><data key="type">SimpleName</data><data key="parentType">ClassOrInterfaceType</data></node><node id="60"><data key="reference"></data><data key="text">{
    return iterator;
}</data><data key="type">BlockStmt</data><data key="parentType">MethodDeclaration</data></node><node id="61"><data key="reference"></data><data key="text">return iterator;</data><data key="type">ReturnStmt</data><data key="parentType">BlockStmt</data></node><node id="62"><data key="reference">org.apache.commons.collections4.MapIterator</data><data key="identifier">iterator</data><data key="text">iterator</data><data key="type">SimpleName</data><data key="parentType">NameExpr</data></node><node id="63"><data key="reference"></data><data key="modifier">public</data><data key="text">// -----------------------------------------------------------------------
/**
 * {@inheritDoc}
 */
public boolean hasNext() {
    return iterator.hasNext();
}</data><data key="type">MethodDeclaration</data><data key="parentType">ClassOrInterfaceDeclaration</data></node><node id="64"><data key="reference">userDefinedMethodName</data><data key="identifier">hasNext</data><data key="text">hasNext</data><data key="type">SimpleName</data><data key="parentType">MethodDeclaration</data></node><node id="65"><data key="reference"></data><data key="identifier">boolean</data><data key="text">boolean</data><data key="type">PrimitiveType</data><data key="parentType">MethodDeclaration</data></node><node id="66"><data key="reference"></data><data key="text">{
    return iterator.hasNext();
}</data><data key="type">BlockStmt</data><data key="parentType">MethodDeclaration</data></node><node id="67"><data key="reference"></data><data key="text">return iterator.hasNext();</data><data key="type">ReturnStmt</data><data key="parentType">BlockStmt</data></node><node id="68"><data key="reference"></data><data key="text">iterator.hasNext()</data><data key="type">MethodCallExpr</data><data key="parentType">ReturnStmt</data></node><node id="69"><data key="reference">org.apache.commons.collections4.MapIterator</data><data key="identifier">iterator</data><data key="text">iterator</data><data key="type">SimpleName</data><data key="parentType">NameExpr</data></node><node id="70"><data key="reference">otherMethodCall</data><data key="identifier">hasNext</data><data key="text">hasNext</data><data key="type">SimpleName</data><data key="parentType">MethodCallExpr</data></node><node id="71"><data key="reference"></data><data key="modifier">public</data><data key="text">/**
 * {@inheritDoc}
 */
public K next() {
    return iterator.next();
}</data><data key="type">MethodDeclaration</data><data key="parentType">ClassOrInterfaceDeclaration</data></node><node id="72"><data key="reference">userDefinedMethodName</data><data key="identifier">next</data><data key="text">next</data><data key="type">SimpleName</data><data key="parentType">MethodDeclaration</data></node><node id="73"><data key="reference"></data><data key="text">K</data><data key="type">ClassOrInterfaceType</data><data key="parentType">MethodDeclaration</data></node><node id="74"><data key="reference">nonQualifiedClassName</data><data key="identifier">K</data><data key="text">K</data><data key="type">SimpleName</data><data key="parentType">ClassOrInterfaceType</data></node><node id="75"><data key="reference"></data><data key="text">{
    return iterator.next();
}</data><data key="type">BlockStmt</data><data key="parentType">MethodDeclaration</data></node><node id="76"><data key="reference"></data><data key="text">return iterator.next();</data><data key="type">ReturnStmt</data><data key="parentType">BlockStmt</data></node><node id="77"><data key="reference"></data><data key="text">iterator.next()</data><data key="type">MethodCallExpr</data><data key="parentType">ReturnStmt</data></node><node id="78"><data key="reference">org.apache.commons.collections4.MapIterator</data><data key="identifier">iterator</data><data key="text">iterator</data><data key="type">SimpleName</data><data key="parentType">NameExpr</data></node><node id="79"><data key="reference">otherMethodCall</data><data key="identifier">next</data><data key="text">next</data><data key="type">SimpleName</data><data key="parentType">MethodCallExpr</data></node><node id="100"><data key="reference"></data><data key="text">{
    return iterator.getValue();
}</data><data key="type">BlockStmt</data><data key="parentType">MethodDeclaration</data></node><node id="101"><data key="reference"></data><data key="text">return iterator.getValue();</data><data key="type">ReturnStmt</data><data key="parentType">BlockStmt</data></node><node id="102"><data key="reference"></data><data key="text">iterator.getValue()</data><data key="type">MethodCallExpr</data><data key="parentType">ReturnStmt</data></node><node id="103"><data key="reference">org.apache.commons.collections4.MapIterator</data><data key="identifier">iterator</data><data key="text">iterator</data><data key="type">SimpleName</data><data key="parentType">NameExpr</data></node><node id="104"><data key="reference">otherMethodCall</data><data key="identifier">getValue</data><data key="text">getValue</data><data key="type">SimpleName</data><data key="parentType">MethodCallExpr</data></node><node id="105"><data key="reference"></data><data key="modifier">public</data><data key="text">/**
 * {@inheritDoc}
 */
public V setValue(final V obj) {
    return iterator.setValue(obj);
}</data><data key="type">MethodDeclaration</data><data key="parentType">ClassOrInterfaceDeclaration</data></node><node id="106"><data key="reference">userDefinedMethodName</data><data key="identifier">setValue</data><data key="text">setValue</data><data key="type">SimpleName</data><data key="parentType">MethodDeclaration</data></node><node id="80"><data key="reference"></data><data key="modifier">public</data><data key="text">/**
 * {@inheritDoc}
 */
public void remove() {
    iterator.remove();
}</data><data key="type">MethodDeclaration</data><data key="parentType">ClassOrInterfaceDeclaration</data></node><node id="107"><data key="reference"></data><data key="modifier">final</data><data key="text">final V obj</data><data key="type">Parameter</data><data key="parentType">MethodDeclaration</data></node><node id="81"><data key="reference">userDefinedMethodName</data><data key="identifier">remove</data><data key="text">remove</data><data key="type">SimpleName</data><data key="parentType">MethodDeclaration</data></node><node id="108"><data key="reference"></data><data key="text">V</data><data key="type">ClassOrInterfaceType</data><data key="parentType">Parameter</data></node><node id="82"><data key="reference"></data><data key="text">void</data><data key="type">VoidType</data><data key="parentType">MethodDeclaration</data></node><node id="109"><data key="reference">nonQualifiedClassName</data><data key="identifier">V</data><data key="text">V</data><data key="type">SimpleName</data><data key="parentType">ClassOrInterfaceType</data></node><node id="83"><data key="reference"></data><data key="text">{
    iterator.remove();
}</data><data key="type">BlockStmt</data><data key="parentType">MethodDeclaration</data></node><node id="84"><data key="reference"></data><data key="text">iterator.remove()</data><data key="type">MethodCallExpr</data><data key="parentType">ExpressionStmt</data></node><node id="85"><data key="reference">org.apache.commons.collections4.MapIterator</data><data key="identifier">iterator</data><data key="text">iterator</data><data key="type">SimpleName</data><data key="parentType">NameExpr</data></node><node id="86"><data key="reference">otherMethodCall</data><data key="identifier">remove</data><data key="text">remove</data><data key="type">SimpleName</data><data key="parentType">MethodCallExpr</data></node><node id="87"><data key="reference"></data><data key="modifier">public</data><data key="text">/**
 * {@inheritDoc}
 */
public K getKey() {
    return iterator.getKey();
}</data><data key="type">MethodDeclaration</data><data key="parentType">ClassOrInterfaceDeclaration</data></node><edge id="131" source="88" target="89" label="NEXT_TOKEN"><data key="type">NEXT_TOKEN</data></edge><edge id="134" source="89" target="90" label="AST"><data key="type">AST</data></edge><edge id="133" source="89" target="91" label="NEXT_TOKEN"><data key="type">NEXT_TOKEN</data></edge><edge id="168" source="111" target="112" label="AST"><data key="type">AST</data></edge><edge id="167" source="111" target="113" label="NEXT_TOKEN"><data key="type">NEXT_TOKEN</data></edge><edge id="170" source="113" target="114" label="AST"><data key="type">AST</data></edge><edge id="171" source="114" target="115" label="AST"><data key="type">AST</data></edge><edge id="180" source="114" target="105" label="RETURNS_TO"><data key="type">RETURNS_TO</data></edge><edge id="176" source="115" target="118" label="AST"><data key="type">AST</data></edge><edge id="172" source="115" target="116" label="AST"><data key="type">AST</data></edge><edge id="174" source="115" target="117" label="AST"><data key="type">AST</data></edge><edge id="173" source="116" target="117" label="NEXT_TOKEN"><data key="type">NEXT_TOKEN</data></edge><edge id="175" source="117" target="118" label="NEXT_TOKEN"><data key="type">NEXT_TOKEN</data></edge><edge id="136" source="91" target="92" label="AST"><data key="type">AST</data></edge><edge id="177" source="118" target="110" label="LAST_WRITE"><data key="type">LAST_WRITE</data></edge><edge id="179" source="118" target="110" label="LAST_LEXICAL_SCOPE_USE"><data key="type">LAST_LEXICAL_SCOPE_USE</data></edge><edge id="178" source="118" target="110" label="LAST_READ"><data key="type">LAST_READ</data></edge><edge id="137" source="92" target="93" label="AST"><data key="type">AST</data></edge><edge id="141" source="92" target="87" label="RETURNS_TO"><data key="type">RETURNS_TO</data></edge><edge id="183" source="119" target="120" label="AST"><data key="type">AST</data></edge><edge id="182" source="119" target="121" label="NEXT_TOKEN"><data key="type">NEXT_TOKEN</data></edge><edge id="138" source="93" target="94" label="AST"><data key="type">AST</data></edge><edge id="140" source="93" target="95" label="AST"><data key="type">AST</data></edge><edge id="139" source="94" target="95" label="NEXT_TOKEN"><data key="type">NEXT_TOKEN</data></edge><edge id="144" source="96" target="97" label="AST"><data key="type">AST</data></edge><edge id="146" source="96" target="98" label="AST"><data key="type">AST</data></edge><edge id="149" source="96" target="100" label="AST"><data key="type">AST</data></edge><edge id="143" source="96" target="105" label="NEXT_TOKEN"><data key="type">NEXT_TOKEN</data></edge><edge id="145" source="97" target="98" label="NEXT_TOKEN"><data key="type">NEXT_TOKEN</data></edge><edge id="11" source="10" target="11" label="AST"><data key="type">AST</data></edge><edge id="148" source="98" target="99" label="AST"><data key="type">AST</data></edge><edge id="147" source="98" target="100" label="NEXT_TOKEN"><data key="type">NEXT_TOKEN</data></edge><edge id="12" source="11" target="12" label="AST"><data key="type">AST</data></edge><edge id="13" source="12" target="13" label="AST"><data key="type">AST</data></edge><edge id="15" source="14" target="15" label="AST"><data key="type">AST</data></edge><edge id="17" source="14" target="16" label="AST"><data key="type">AST</data></edge><edge id="181" source="14" target="119" label="AST"><data key="type">AST</data></edge><edge id="184" source="14" target="121" label="AST"><data key="type">AST</data></edge><edge id="142" source="14" target="96" label="AST"><data key="type">AST</data></edge><edge id="187" source="14" target="123" label="AST"><data key="type">AST</data></edge><edge id="156" source="14" target="105" label="AST"><data key="type">AST</data></edge><edge id="103" source="14" target="71" label="AST"><data key="type">AST</data></edge><edge id="90" source="14" target="63" label="AST"><data key="type">AST</data></edge><edge id="117" source="14" target="80" label="AST"><data key="type">AST</data></edge><edge id="128" source="14" target="87" label="AST"><data key="type">AST</data></edge><edge id="73" source="14" target="52" label="AST"><data key="type">AST</data></edge><edge id="30" source="14" target="25" label="AST"><data key="type">AST</data></edge><edge id="16" source="15" target="16" label="NEXT_TOKEN"><data key="type">NEXT_TOKEN</data></edge><edge id="19" source="16" target="17" label="AST"><data key="type">AST</data></edge><edge id="18" source="16" target="25" label="NEXT_TOKEN"><data key="type">NEXT_TOKEN</data></edge><edge id="29" source="17" target="24" label="AST"><data key="type">AST</data></edge><edge id="20" source="17" target="18" label="AST"><data key="type">AST</data></edge><edge id="22" source="18" target="19" label="AST"><data key="type">AST</data></edge><edge id="24" source="18" target="20" label="AST"><data key="type">AST</data></edge><edge id="27" source="18" target="22" label="AST"><data key="type">AST</data></edge><edge id="21" source="18" target="24" label="NEXT_TOKEN"><data key="type">NEXT_TOKEN</data></edge><edge id="23" source="19" target="20" label="NEXT_TOKEN"><data key="type">NEXT_TOKEN</data></edge><edge id="186" source="121" target="122" label="AST"><data key="type">AST</data></edge><edge id="185" source="121" target="123" label="NEXT_TOKEN"><data key="type">NEXT_TOKEN</data></edge><edge id="0" source="1" target="2" label="AST"><data key="type">AST</data></edge><edge id="14" source="1" target="14" label="AST"><data key="type">AST</data></edge><edge id="7" source="1" target="8" label="AST"><data key="type">AST</data></edge><edge id="2" source="2" target="3" label="AST"><data key="type">AST</data></edge><edge id="1" source="2" target="8" label="NEXT_TOKEN"><data key="type">NEXT_TOKEN</data></edge><edge id="188" source="123" target="124" label="AST"><data key="type">AST</data></edge><edge id="190" source="123" target="125" label="AST"><data key="type">AST</data></edge><edge id="193" source="123" target="127" label="AST"><data key="type">AST</data></edge><edge id="3" source="3" target="4" label="AST"><data key="type">AST</data></edge><edge id="189" source="124" target="125" label="NEXT_TOKEN"><data key="type">NEXT_TOKEN</data></edge><edge id="4" source="4" target="5" label="AST"><data key="type">AST</data></edge><edge id="192" source="125" target="126" label="AST"><data key="type">AST</data></edge><edge id="191" source="125" target="127" label="NEXT_TOKEN"><data key="type">NEXT_TOKEN</data></edge><edge id="5" source="5" target="6" label="AST"><data key="type">AST</data></edge><edge id="6" source="6" target="7" label="AST"><data key="type">AST</data></edge><edge id="194" source="127" target="128" label="AST"><data key="type">AST</data></edge><edge id="9" source="8" target="9" label="AST"><data key="type">AST</data></edge><edge id="8" source="8" target="14" label="NEXT_TOKEN"><data key="type">NEXT_TOKEN</data></edge><edge id="10" source="9" target="10" label="AST"><data key="type">AST</data></edge><edge id="26" source="20" target="21" label="AST"><data key="type">AST</data></edge><edge id="25" source="20" target="22" label="NEXT_TOKEN"><data key="type">NEXT_TOKEN</data></edge><edge id="28" source="22" target="23" label="AST"><data key="type">AST</data></edge><edge id="34" source="25" target="27" label="AST"><data key="type">AST</data></edge><edge id="46" source="25" target="35" label="AST"><data key="type">AST</data></edge><edge id="32" source="25" target="26" label="AST"><data key="type">AST</data></edge><edge id="31" source="25" target="52" label="NEXT_TOKEN"><data key="type">NEXT_TOKEN</data></edge><edge id="33" source="26" target="27" label="NEXT_TOKEN"><data key="type">NEXT_TOKEN</data></edge><edge id="45" source="27" target="34" label="AST"><data key="type">AST</data></edge><edge id="36" source="27" target="28" label="AST"><data key="type">AST</data></edge><edge id="35" source="27" target="35" label="NEXT_TOKEN"><data key="type">NEXT_TOKEN</data></edge><edge id="38" source="28" target="29" label="AST"><data key="type">AST</data></edge><edge id="40" source="28" target="30" label="AST"><data key="type">AST</data></edge><edge id="43" source="28" target="32" label="AST"><data key="type">AST</data></edge><edge id="37" source="28" target="34" label="NEXT_TOKEN"><data key="type">NEXT_TOKEN</data></edge><edge id="39" source="29" target="30" label="NEXT_TOKEN"><data key="type">NEXT_TOKEN</data></edge><edge id="42" source="30" target="31" label="AST"><data key="type">AST</data></edge><edge id="41" source="30" target="32" label="NEXT_TOKEN"><data key="type">NEXT_TOKEN</data></edge><edge id="44" source="32" target="33" label="AST"><data key="type">AST</data></edge><edge id="47" source="35" target="36" label="AST"><data key="type">AST</data></edge><edge id="49" source="35" target="37" label="AST"><data key="type">AST</data></edge><edge id="63" source="35" target="47" label="AST"><data key="type">AST</data></edge><edge id="48" source="36" target="37" label="NEXT_TOKEN"><data key="type">NEXT_TOKEN</data></edge><edge id="56" source="37" target="41" label="AST"><data key="type">AST</data></edge><edge id="51" source="37" target="38" label="AST"><data key="type">AST</data></edge><edge id="50" source="37" target="47" label="NEXT_TOKEN"><data key="type">NEXT_TOKEN</data></edge><edge id="55" source="38" target="40" label="AST"><data key="type">AST</data></edge><edge id="53" source="38" target="39" label="AST"><data key="type">AST</data></edge><edge id="52" source="38" target="41" label="NEXT_TOKEN"><data key="type">NEXT_TOKEN</data></edge><edge id="54" source="39" target="40" label="NEXT_TOKEN"><data key="type">NEXT_TOKEN</data></edge><edge id="57" source="41" target="42" label="AST"><data key="type">AST</data></edge><edge id="58" source="42" target="43" label="AST"><data key="type">AST</data></edge><edge id="59" source="43" target="44" label="AST"><data key="type">AST</data></edge><edge id="62" source="43" target="46" label="AST"><data key="type">AST</data></edge><edge id="61" source="44" target="45" label="AST"><data key="type">AST</data></edge><edge id="60" source="44" target="46" label="NEXT_TOKEN"><data key="type">NEXT_TOKEN</data></edge><edge id="72" source="47" target="51" label="AST"><data key="type">AST</data></edge><edge id="64" source="47" target="48" label="AST"><data key="type">AST</data></edge><edge id="66" source="48" target="49" label="AST"><data key="type">AST</data></edge><edge id="68" source="48" target="50" label="AST"><data key="type">AST</data></edge><edge id="71" source="48" target="17" label="LAST_FIELD_LEX"><data key="type">LAST_FIELD_LEX</data></edge><edge id="65" source="48" target="51" label="NEXT_TOKEN"><data key="type">NEXT_TOKEN</data></edge><edge id="67" source="49" target="50" label="NEXT_TOKEN"><data key="type">NEXT_TOKEN</data></edge><edge id="70" source="50" target="51" label="COMPUTED_FROM"><data key="type">COMPUTED_FROM</data></edge><edge id="69" source="50" target="24" label="FIELD"><data key="type">FIELD</data></edge><edge id="77" source="52" target="54" label="AST"><data key="type">AST</data></edge><edge id="75" source="52" target="53" label="AST"><data key="type">AST</data></edge><edge id="86" source="52" target="60" label="AST"><data key="type">AST</data></edge><edge id="74" source="52" target="63" label="NEXT_TOKEN"><data key="type">NEXT_TOKEN</data></edge><edge id="76" source="53" target="54" label="NEXT_TOKEN"><data key="type">NEXT_TOKEN</data></edge><edge id="79" source="54" target="55" label="AST"><data key="type">AST</data></edge><edge id="81" source="54" target="56" label="AST"><data key="type">AST</data></edge><edge id="84" source="54" target="58" label="AST"><data key="type">AST</data></edge><edge id="78" source="54" target="60" label="NEXT_TOKEN"><data key="type">NEXT_TOKEN</data></edge><edge id="80" source="55" target="56" label="NEXT_TOKEN"><data key="type">NEXT_TOKEN</data></edge><edge id="83" source="56" target="57" label="AST"><data key="type">AST</data></edge><edge id="82" source="56" target="58" label="NEXT_TOKEN"><data key="type">NEXT_TOKEN</data></edge><edge id="85" source="58" target="59" label="AST"><data key="type">AST</data></edge><edge id="87" source="60" target="61" label="AST"><data key="type">AST</data></edge><edge id="88" source="61" target="62" label="AST"><data key="type">AST</data></edge><edge id="89" source="61" target="52" label="RETURNS_TO"><data key="type">RETURNS_TO</data></edge><edge id="92" source="63" target="64" label="AST"><data key="type">AST</data></edge><edge id="94" source="63" target="65" label="AST"><data key="type">AST</data></edge><edge id="96" source="63" target="66" label="AST"><data key="type">AST</data></edge><edge id="91" source="63" target="71" label="NEXT_TOKEN"><data key="type">NEXT_TOKEN</data></edge><edge id="93" source="64" target="65" label="NEXT_TOKEN"><data key="type">NEXT_TOKEN</data></edge><edge id="95" source="65" target="66" label="NEXT_TOKEN"><data key="type">NEXT_TOKEN</data></edge><edge id="97" source="66" target="67" label="AST"><data key="type">AST</data></edge><edge id="98" source="67" target="68" label="AST"><data key="type">AST</data></edge><edge id="102" source="67" target="63" label="RETURNS_TO"><data key="type">RETURNS_TO</data></edge><edge id="99" source="68" target="69" label="AST"><data key="type">AST</data></edge><edge id="101" source="68" target="70" label="AST"><data key="type">AST</data></edge><edge id="100" source="69" target="70" label="NEXT_TOKEN"><data key="type">NEXT_TOKEN</data></edge><edge id="110" source="71" target="75" label="AST"><data key="type">AST</data></edge><edge id="105" source="71" target="72" label="AST"><data key="type">AST</data></edge><edge id="107" source="71" target="73" label="AST"><data key="type">AST</data></edge><edge id="104" source="71" target="80" label="NEXT_TOKEN"><data key="type">NEXT_TOKEN</data></edge><edge id="106" source="72" target="73" label="NEXT_TOKEN"><data key="type">NEXT_TOKEN</data></edge><edge id="109" source="73" target="74" label="AST"><data key="type">AST</data></edge><edge id="108" source="73" target="75" label="NEXT_TOKEN"><data key="type">NEXT_TOKEN</data></edge><edge id="111" source="75" target="76" label="AST"><data key="type">AST</data></edge><edge id="112" source="76" target="77" label="AST"><data key="type">AST</data></edge><edge id="116" source="76" target="71" label="RETURNS_TO"><data key="type">RETURNS_TO</data></edge><edge id="113" source="77" target="78" label="AST"><data key="type">AST</data></edge><edge id="115" source="77" target="79" label="AST"><data key="type">AST</data></edge><edge id="114" source="78" target="79" label="NEXT_TOKEN"><data key="type">NEXT_TOKEN</data></edge><edge id="150" source="100" target="101" label="AST"><data key="type">AST</data></edge><edge id="151" source="101" target="102" label="AST"><data key="type">AST</data></edge><edge id="155" source="101" target="96" label="RETURNS_TO"><data key="type">RETURNS_TO</data></edge><edge id="154" source="102" target="104" label="AST"><data key="type">AST</data></edge><edge id="152" source="102" target="103" label="AST"><data key="type">AST</data></edge><edge id="153" source="103" target="104" label="NEXT_TOKEN"><data key="type">NEXT_TOKEN</data></edge><edge id="166" source="105" target="111" label="AST"><data key="type">AST</data></edge><edge id="158" source="105" target="106" label="AST"><data key="type">AST</data></edge><edge id="169" source="105" target="113" label="AST"><data key="type">AST</data></edge><edge id="160" source="105" target="107" label="AST"><data key="type">AST</data></edge><edge id="157" source="105" target="119" label="NEXT_TOKEN"><data key="type">NEXT_TOKEN</data></edge><edge id="159" source="106" target="107" label="NEXT_TOKEN"><data key="type">NEXT_TOKEN</data></edge><edge id="121" source="80" target="82" label="AST"><data key="type">AST</data></edge><edge id="123" source="80" target="83" label="AST"><data key="type">AST</data></edge><edge id="119" source="80" target="81" label="AST"><data key="type">AST</data></edge><edge id="118" source="80" target="87" label="NEXT_TOKEN"><data key="type">NEXT_TOKEN</data></edge><edge id="165" source="107" target="110" label="AST"><data key="type">AST</data></edge><edge id="162" source="107" target="108" label="AST"><data key="type">AST</data></edge><edge id="161" source="107" target="111" label="NEXT_TOKEN"><data key="type">NEXT_TOKEN</data></edge><edge id="120" source="81" target="82" label="NEXT_TOKEN"><data key="type">NEXT_TOKEN</data></edge><edge id="164" source="108" target="109" label="AST"><data key="type">AST</data></edge><edge id="163" source="108" target="110" label="NEXT_TOKEN"><data key="type">NEXT_TOKEN</data></edge><edge id="122" source="82" target="83" label="NEXT_TOKEN"><data key="type">NEXT_TOKEN</data></edge><edge id="124" source="83" target="84" label="AST"><data key="type">AST</data></edge><edge id="125" source="84" target="85" label="AST"><data key="type">AST</data></edge><edge id="127" source="84" target="86" label="AST"><data key="type">AST</data></edge><edge id="126" source="85" target="86" label="NEXT_TOKEN"><data key="type">NEXT_TOKEN</data></edge><edge id="132" source="87" target="89" label="AST"><data key="type">AST</data></edge><edge id="135" source="87" target="91" label="AST"><data key="type">AST</data></edge><edge id="130" source="87" target="88" label="AST"><data key="type">AST</data></edge><edge id="129" source="87" target="96" label="NEXT_TOKEN"><data key="type">NEXT_TOKEN</data></edge></graph></graphml>