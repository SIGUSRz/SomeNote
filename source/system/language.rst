##############
Language
##############

Table of Contents
*****************

.. contents::

String
*******

Java
====

.. hint:: **String Creation**

    - :code:`=` 
        - if string already exists, assignment will give the same address to the variable, so variable is just another **reference** of the same string
        - Example: :code:`sa` and :code:`sb` are referencing same same address
        - Java String is **immutable**, changes like :code:`+` and :code:`.replace()` on string creates new copy

    .. code-block:: Java

        public static void main(String argv[]) {
            String sa = "abc";
            String sb = "abc";
            if (sa == sb) {
                System.out.println("Yes");
            } else {
                System.out.println("No");
            }
        } //return Yes

    - Empty String
        - :code:`String s = ""`
        - Can perform all string operations


.. hint:: **String Traverse**

    - :code:`s.length()` get length
    - :code:`s.charAt(i)` get char at i

.. hint:: **String Comparison**

    - Java :guilabel:`String` is a Class
    - Use :code:`==`: the operator compares between **instance memory address** and **content value**
    - :code:`.equals()`
        - Use :code:`.equals()`: the function compares between content value
        - Java String rewrite :code:`.equals()` so that different address but same content return true
        - Java StringBuilder do not rewrite :code:`.equals()` so that only same address and same content return true

.. hint:: **String Operations**

    - Remove space
        - :code:`String trim = s.trim();`
    - Concat
        - :code:`String concat = s.concat("concat");`
    - Char Array
        - :code:`char[] charArray = s.toCharArray();`
    - Case
        - :code:`String upper = s.toUpperCase();`

Python
======

.. hint:: **String Creation**

    - :code:`=`
        - assignment: :code:`s = "abc"`
        - immutable: :code:`s[i] = x` do not work
    - Empty String
        - :code:`s = ""`
        - :code:`s = str()`
        - Can perform all string operations

.. hint:: **String Traverse**

    - :code:`len(s)` get length
    - :code:`s[i]` get char at i

.. hint:: **String Comparison**

    - Use :code:`==` to compare string content

C++
====

.. hint:: **String Creation**

    - :code:`=`
        - assignment: :code:`s = "abc"`
        - mutable: :code:`s[i] = x`
    - Empty String
        - :code:`string s;`
        - Can perform all string operations

Memory (内存空间)
********************

Stack Space (栈空间)
=======================

.. hint::
    Stack Space

    - OS preallocate a fixed space for a single process isolated from other processes.
    - The process performs function call, recurssion in the stack space
    - Stack Space Storage
        - Function parameters and returned values
        - Function local variables
            - Array name: a local variable pointing to actual array address, stored in **stack space**
            - Values in array: actual array values that are pointed by array name local variable, stored in **heap space**
    - Values in stack space are **cleared** when function ends and process terminates.