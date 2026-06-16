#lang racket
(define (do_algebra operator operand)
  (define (op-fn o)
    (cond [(string=? o "+") +] [(string=? o "-") -] [(string=? o "*") *]
          [(string=? o "//") quotient] [(string=? o "**") expt]))
  (define (reduce nums ops matchset)
    (let loop ([ns nums] [os ops] [accn '()] [acco '()])
      (cond [(null? os) (values (reverse (cons (car ns) accn)) (reverse acco))]
            [(member (car os) matchset)
             (loop (cons ((op-fn (car os)) (car ns) (cadr ns)) (cddr ns)) (cdr os) accn acco)]
            [else (loop (cdr ns) (cdr os) (cons (car ns) accn) (cons (car os) acco))])))
  (define-values (n1 o1) (reduce operand operator '("**")))
  (define-values (n2 o2) (reduce n1 o1 '("*" "//")))
  (define-values (n3 o3) (reduce n2 o2 '("+" "-")))
  (car n3))
