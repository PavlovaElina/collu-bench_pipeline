#lang racket
(define (order_by_points nums)
  (define (dsum n)
    (define digs (map (lambda (c) (- (char->integer c) 48)) (string->list (number->string (abs n)))))
    (if (< n 0) (apply + (cons (- (car digs)) (cdr digs))) (apply + digs)))
  (map cdr (sort (map (lambda (x i) (cons i x)) nums (range (length nums)))
                 (lambda (a b)
                   (define sa (dsum (cdr a))) (define sb (dsum (cdr b)))
                   (cond [(< sa sb) #t] [(> sa sb) #f] [else (< (car a) (car b))])))))
