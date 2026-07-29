import org.apache.jena.query.Query;
import org.apache.jena.sparql.algebra.Algebra;
import org.apache.jena.sparql.algebra.Op;
import org.apache.jena.sparql.algebra.OpAsQuery;
import org.apache.jena.sparql.sse.SSE;

public class SExprToSparql {
    public static void main(String[] args) throws Exception {
        if (args.length != 1) {
            System.err.println("Usage: java SExprToSparql \"<SEXPR>\"");
            System.exit(1);
        }

        String sexpr = args[0];

        // Parse s-expression back to algebra Op
        Op op = SSE.parseOp(sexpr);

        // Convert algebra back to Query object
        Query query = OpAsQuery.asQuery(op);

        // Serialize to SPARQL string
        System.out.println(query.serialize());
    }
}