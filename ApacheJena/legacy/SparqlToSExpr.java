import org.apache.jena.query.Query;
import org.apache.jena.query.QueryFactory;
import org.apache.jena.sparql.algebra.Algebra;
import org.apache.jena.sparql.algebra.Op;
import org.apache.jena.sparql.sse.SSE;

public class SparqlToSExpr {
    public static void main(String[] args) throws Exception {
        if (args.length != 1) {
            System.err.println("Usage: java SparqlToSExpr \"<SPARQL_QUERY>\"");
            System.exit(1);
        }

        String sparqlQuery = args[0];

        // Parse query
        Query query = QueryFactory.create(sparqlQuery);

        // Compile to algebra
        Op op = Algebra.compile(query);

        // Print S-expression
        SSE.write(System.out, op);
    }
}
