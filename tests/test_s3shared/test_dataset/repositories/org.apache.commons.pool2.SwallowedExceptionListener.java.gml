<?xml version="1.0" ?><graphml xmlns="http://graphml.graphdrawing.org/xmlns" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="http://graphml.graphdrawing.org/xmlns http://graphml.graphdrawing.org/xmlns/1.1/graphml.xsd"><key id="reference" for="node" attr.name="reference" attr.type="string"></key><key id="identifier" for="node" attr.name="identifier" attr.type="string"></key><key id="modifier" for="node" attr.name="modifier" attr.type="string"></key><key id="text" for="node" attr.name="text" attr.type="string"></key><key id="type" for="node" attr.name="type" attr.type="string"></key><key id="parentType" for="node" attr.name="parentType" attr.type="string"></key><key id="type" for="edge" attr.name="type" attr.type="string"></key><graph id="G" edgedefault="directed"><node id="11"><data key="reference"></data><data key="modifier"></data><data key="text">Exception e</data><data key="type">Parameter</data><data key="parentType">MethodDeclaration</data></node><node id="12"><data key="reference"></data><data key="text">Exception</data><data key="type">ClassOrInterfaceType</data><data key="parentType">Parameter</data></node><node id="13"><data key="reference">nonQualifiedClassName</data><data key="identifier">Exception</data><data key="text">Exception</data><data key="type">SimpleName</data><data key="parentType">ClassOrInterfaceType</data></node><node id="14"><data key="reference">java.lang.Exception</data><data key="identifier">e</data><data key="text">e</data><data key="type">SimpleName</data><data key="parentType">Parameter</data></node><node id="15"><data key="reference"></data><data key="text">void</data><data key="type">VoidType</data><data key="parentType">MethodDeclaration</data></node><node id="1"><data key="reference"></data><data key="text">/*
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
package org.apache.commons.pool2;

/**
 * Pools that unavoidably swallow exceptions may be configured with an instance
 * of this listener so the user may receive notification of when this happens.
 * The listener should not throw an exception when called but pools calling
 * listeners should protect themselves against exceptions anyway.
 *
 * @since 2.0
 */
public interface SwallowedExceptionListener {

    /**
     * This method is called every time the implementation unavoidably swallows
     * an exception.
     *
     * @param e The exception that was swallowed
     */
    void onSwallowException(Exception e);
}
</data><data key="type">CompilationUnit</data></node><node id="2"><data key="reference"></data><data key="text">package org.apache.commons.pool2;

</data><data key="type">PackageDeclaration</data><data key="parentType">CompilationUnit</data></node><node id="3"><data key="reference"></data><data key="identifier">pool2</data><data key="text">org.apache.commons.pool2</data><data key="type">Name</data><data key="parentType">PackageDeclaration</data></node><node id="4"><data key="reference"></data><data key="identifier">commons</data><data key="text">org.apache.commons</data><data key="type">Name</data><data key="parentType">Name</data></node><node id="5"><data key="reference"></data><data key="identifier">apache</data><data key="text">org.apache</data><data key="type">Name</data><data key="parentType">Name</data></node><node id="6"><data key="reference"></data><data key="identifier">org</data><data key="text">org</data><data key="type">Name</data><data key="parentType">Name</data></node><node id="7"><data key="reference"></data><data key="modifier">public,interface</data><data key="text">/**
 * Pools that unavoidably swallow exceptions may be configured with an instance
 * of this listener so the user may receive notification of when this happens.
 * The listener should not throw an exception when called but pools calling
 * listeners should protect themselves against exceptions anyway.
 *
 * @since 2.0
 */
public interface SwallowedExceptionListener {

    /**
     * This method is called every time the implementation unavoidably swallows
     * an exception.
     *
     * @param e The exception that was swallowed
     */
    void onSwallowException(Exception e);
}</data><data key="type">ClassOrInterfaceDeclaration</data><data key="parentType">CompilationUnit</data></node><node id="8"><data key="reference"></data><data key="identifier">SwallowedExceptionListener</data><data key="text">SwallowedExceptionListener</data><data key="type">SimpleName</data><data key="parentType">ClassOrInterfaceDeclaration</data></node><node id="9"><data key="reference"></data><data key="modifier"></data><data key="text">/**
 * This method is called every time the implementation unavoidably swallows
 * an exception.
 *
 * @param e The exception that was swallowed
 */
void onSwallowException(Exception e);</data><data key="type">MethodDeclaration</data><data key="parentType">ClassOrInterfaceDeclaration</data></node><node id="10"><data key="reference">userDefinedMethodName</data><data key="identifier">onSwallowException</data><data key="text">onSwallowException</data><data key="type">SimpleName</data><data key="parentType">MethodDeclaration</data></node><edge id="14" source="11" target="12" label="AST"><data key="type">AST</data></edge><edge id="17" source="11" target="14" label="AST"><data key="type">AST</data></edge><edge id="13" source="11" target="15" label="NEXT_TOKEN"><data key="type">NEXT_TOKEN</data></edge><edge id="16" source="12" target="13" label="AST"><data key="type">AST</data></edge><edge id="15" source="12" target="14" label="NEXT_TOKEN"><data key="type">NEXT_TOKEN</data></edge><edge id="0" source="1" target="2" label="AST"><data key="type">AST</data></edge><edge id="6" source="1" target="7" label="AST"><data key="type">AST</data></edge><edge id="2" source="2" target="3" label="AST"><data key="type">AST</data></edge><edge id="1" source="2" target="7" label="NEXT_TOKEN"><data key="type">NEXT_TOKEN</data></edge><edge id="3" source="3" target="4" label="AST"><data key="type">AST</data></edge><edge id="4" source="4" target="5" label="AST"><data key="type">AST</data></edge><edge id="5" source="5" target="6" label="AST"><data key="type">AST</data></edge><edge id="7" source="7" target="8" label="AST"><data key="type">AST</data></edge><edge id="9" source="7" target="9" label="AST"><data key="type">AST</data></edge><edge id="8" source="8" target="9" label="NEXT_TOKEN"><data key="type">NEXT_TOKEN</data></edge><edge id="12" source="9" target="11" label="AST"><data key="type">AST</data></edge><edge id="18" source="9" target="15" label="AST"><data key="type">AST</data></edge><edge id="10" source="9" target="10" label="AST"><data key="type">AST</data></edge><edge id="11" source="10" target="11" label="NEXT_TOKEN"><data key="type">NEXT_TOKEN</data></edge></graph></graphml>