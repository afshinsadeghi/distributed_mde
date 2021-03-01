mvn clean install package -Dskip -Dmaven.test.skip=true -DskipTests=true -Dmaven.javadoc.skip=true
mvn exec:java -Dexec.mainClass="MDE"
